# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks import Hook

from xtuner.registry import BUILDER
from xtuner.utils import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX


def split_list(lst, value):
    res = []
    tmp_res = []
    for i in lst:
        # if tmp_res and i == value:
        if i == value:
            res.append(tmp_res)
            tmp_res = []
        else:
            tmp_res.append(i)
    res.append(tmp_res)
    return res


class DatasetInfoHook(Hook):

    def __init__(self, tokenizer, is_intern_repo_dataset=False):
        # self.tokenizer = BUILDER.build(tokenizer)

        import time
        max_times = 10
        for _ in range(max_times):
            try:
                self.tokenizer = BUILDER.build(tokenizer)
                break
            except Exception as e:
                print('\n build tokenizer errors, will try again')
                time.sleep(10)

        self.is_intern_repo_dataset = is_intern_repo_dataset

    def log(self, runner, dataset, mode='train'):
        runner.logger.info(f'Num {mode} samples {len(dataset)}')
        runner.logger.info(f'{mode} example:')

        if 'input_ids' in dataset[0]:
            input_ids = dataset[0]['input_ids']
            if self.is_intern_repo_dataset:
                input_ids = [abs(x) for x in input_ids]
            # Try to split list to be compatible with IMAGE token
            input_ids = split_list(input_ids, IMAGE_TOKEN_INDEX)
            text = ''
            for idx, ids in enumerate(input_ids):
                try:
                    if len(ids) != 0:
                        text += self.tokenizer.decode(ids)
                except Exception as e:
                    print(f"{'-'*50}\nerror {e}\n ids: {ids}\n{'-'*50}")
                if idx != len(input_ids) - 1:
                    text += DEFAULT_IMAGE_TOKEN
            runner.logger.info(text)
        else:
            runner.logger.info(dataset[0]['conversations'])


    def log_iter_data(self, runner, train_dataloader, mode='train', iter=0):
        for i, item in enumerate(train_dataloader):
            if i == iter:
                runner.logger.info(f"{len(item['data'])} data in iter {i}: ")
                for input_ids in item['data']['input_ids']:
                    input_ids = split_list(input_ids, IMAGE_TOKEN_INDEX)
                    text = ''
                    for idx, ids in enumerate(input_ids):
                        try:
                            if len(ids) != 0:
                                text += self.tokenizer.decode(ids)
                        except Exception as e:
                            print(f"{'-'*50}\nerror {e}\n ids: {ids}\n{'-'*50}")
                        if idx != len(input_ids) - 1:
                            text += DEFAULT_IMAGE_TOKEN
                    text = text.replace('<unk>', '')
                    runner.logger.info(f"\n{'=' * 50}\n{text}{'=' * 50}")
                break
            

    def before_train(self, runner) -> None:
        do_train = runner.train_loop is not None
        do_eval = runner.val_loop is not None
        do_test = runner.test_loop is not None
        if do_train:
            train_dataloader = runner.train_dataloader
            train_dataset = runner.train_dataloader.dataset
            self.log(runner, train_dataset, mode='train')
            # self.log_iter_data(runner, train_dataloader, mode='train')
        if do_eval:
            eval_dataset = runner.val_dataloader.dataset
            self.log(runner, eval_dataset, mode='eval')
        if do_test:
            test_dataset = runner.test_dataloader.dataset
            self.log(runner, test_dataset, mode='test')

    def before_val(self, runner) -> None:
        eval_dataset = runner.val_dataloader.dataset
        self.log(runner, eval_dataset, mode='eval')

    def before_test(self, runner) -> None:
        test_dataset = runner.test_dataloader.dataset
        self.log(runner, test_dataset, mode='test')
