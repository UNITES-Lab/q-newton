# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2023/11/26
import time
from itertools import chain
from typing import Optional

import torch
import wandb
from datasets import load_dataset, enable_caching
from fire import Fire
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    BertForMaskedLM,
    BertConfig,
    BertTokenizerFast,
    DataCollatorForLanguageModeling,
    get_scheduler,
    set_seed,
)

from models.utils import load_and_freeze_pretrained_embedding_for_bert
from newton_optim import (
    add_hooks_,
    disable_hooks_,
    enable_hooks_,
    clear_backprops_,
    rescale_gradient_for_all_,
)

set_seed(233)

enable_caching()


def group_texts(examples, block_size=512):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result


def train_bert_tiny_from_scratch(
        pretrained_embedding_name_or_path: str,
        enable_newton: Optional[bool] = True,
        profile_hessian_per_steps: Optional[int] = 0,
        delay_newton_steps: Optional[int] = 0,
        # training arguments
        batch_size: Optional[int] = 256,
        num_epochs: Optional[int] = 10,
        learning_rate: Optional[float] = 1e-4,
        weight_decay: Optional[float] = 0.01,
        warmup_steps: Optional[int] = 100,
        evaluation_steps: Optional[int] = 100,
        # model arguments
        hidden_size: Optional[int] = 128,
        num_hidden_layers: Optional[int] = 4,
        num_attention_heads: Optional[int] = 4,
        intermediate_size: Optional[int] = 256,
        max_position_embeddings: Optional[int] = 512,
        quantum_sparsity_ratio: Optional[float] = 1.0,
        normalization_epsilon: Optional[float] = 1,
):
    if not enable_newton:
        delay_newton_steps = -1
    # load dataset & tokenizer
    dataset = load_dataset('wikitext', 'wikitext-103-v1')
    # dataset = load_dataset('JeanKaddour/minipile')
    tokenizer = BertTokenizerFast.from_pretrained(pretrained_embedding_name_or_path)
    # remove empty lines
    dataset = dataset.map(
        lambda x: {'text': [line for line in x['text'] if len(line) > 0 and not line.isspace()]},
        batched=True,
        num_proc=4,
        desc='Remove empty lines',
    )
    dataset = dataset.map(
        lambda x: tokenizer(x['text'], return_special_tokens_mask=True, truncation=False),
        batched=True,
        num_proc=4,
        remove_columns=['text'],
        desc='Tokenize',
    )
    dataset = dataset.map(
        group_texts,
        batched=True,
        num_proc=4,
        desc='Group texts',
    )

    dataset.set_format(type='torch', columns=['input_ids', 'special_tokens_mask'])
    train_set, eval_set = dataset['train'], dataset['validation']

    # data loader
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=0.15,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size, shuffle=True, collate_fn=data_collator
    )
    eval_loader = DataLoader(
        eval_set, batch_size=batch_size, shuffle=False, collate_fn=data_collator
    )

    # model
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings,
        type_vocab_size=2,
    )
    model = BertForMaskedLM(config=config)
    model = load_and_freeze_pretrained_embedding_for_bert(
        model,
        pretrained_model_name_or_path=pretrained_embedding_name_or_path
    )

    # optimizer
    optimizer = SGD(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    ) if enable_newton else SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=len(train_set) // batch_size * num_epochs,
    )

    wandb.init(
        project='qnewton',
        name='train-bert-tiny-from-scratch' + ('-with-newton' if enable_newton else '') + f'-{normalization_epsilon}',
    )

    print(f"Total number of parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Total number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"Total training steps: {len(train_set) // batch_size * num_epochs}")
    print(f"Total epochs: {num_epochs}")
    print(f"Warmup steps: {warmup_steps}")

    # train!
    best_eval_loss = float('inf')
    if enable_newton:
        has_hooks = True
        add_hooks_(model)
        print('Enable Newton')
    else:
        has_hooks = False
    model.train()
    model = model.cuda()
    total_step = 0
    progress_bar = tqdm(range(num_epochs * len(train_loader)), desc='Training')
    timeit_total = {'forward-backward': 0, 'hessian': 0, 'inversion': 0, 'rescale': 0}
    for epoch in range(num_epochs):
        for batch in train_loader:
            batch = {k: v.cuda() for k, v in batch.items()}
            enable_newton = (total_step >= delay_newton_steps) and (delay_newton_steps >= 0)
            if has_hooks:
                clear_backprops_(model)
            start = time.time()
            torch.cuda.synchronize()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.cuda.synchronize()
            end = time.time()
            # progress_bar.write(f'[timeit] forward + gradient backward: {round(end - start, 3)} s')
            timeit_total['forward-backward'] += end - start

            # newton
            if enable_newton:
                step_timeit_dict = rescale_gradient_for_all_(
                    model,
                    damping=1e-10,
                    normalization_epsilon=normalization_epsilon,
                    max_number_of_params=(hidden_size * intermediate_size * 4 + hidden_size * 4),
                    disable_warning=(total_step != 0),
                    print_fn=lambda x: progress_bar.write(f"[step {total_step}]" + x),
                    profile_hessian=(profile_hessian_per_steps > 0 and total_step % profile_hessian_per_steps == 0),
                    quantum_sparsity_ratio=quantum_sparsity_ratio,
                )
                for k, v in step_timeit_dict.items():
                    timeit_total[k] += v

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            wandb.log({
                'train_loss': loss.item(),
                'learning_rate': scheduler.get_last_lr()[0],
            })
            total_step += 1
            progress_bar.update(1)
            progress_bar.set_postfix(
                {'loss': loss.item(),
                 'learning_rate': scheduler.get_last_lr()[0],
                 }.update({k: round(v, 3) for k, v in timeit_total.items()}))

            # wandb.log({
            #     "time-" + k: v / total_step for k, v in timeit_total.items()
            # }, step=total_step)

            if (total_step + 1) % evaluation_steps == 0:
                model.eval()
                disable_hooks_()
                eval_loss = 0
                for eval_batch in eval_loader:
                    eval_batch = {k: v.cuda() for k, v in eval_batch.items()}
                    eval_outputs = model(**eval_batch)
                    eval_loss += eval_outputs.loss.item()
                eval_loss /= len(eval_loader)

                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                wandb.log({'eval_loss': eval_loss})
                wandb.summary['best_eval_loss'] = best_eval_loss
                model.train()
                enable_hooks_()


if __name__ == '__main__':
    Fire(train_bert_tiny_from_scratch)
