# coding=utf-8
#
# Copyright 2020 Heinrich Heine University Duesseldorf
#
# Part of this code is based on the source code of BERT-DST
# (arXiv:1907.03040)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json

import dataset_woz2
import dataset_sim
import dataset_multiwoz21


class DataProcessor(object):
    def __init__(self, dataset_config):
        with open(dataset_config, "r", encoding='utf-8') as f:
            raw_config = json.load(f)
        self.class_types = raw_config['class_types']
        self.slot_list = raw_config['slots']
        self.domain_list = []
        one_domain = False
        for slot in self.slot_list:
            token = slot.split("-")
            if len(token) == 2:
                domain, _ = slot.split("-")
                self.domain_list.append(domain)
            else:
                one_domain = True
        if one_domain:
            self.domain_list.append("root")
        self.domain_list = list(set(self.domain_list))
        self.label_maps = raw_config['label_maps']
        self.occur_list = raw_config['occur_list']

    def get_train_examples(self, data_dir, **args):
        raise NotImplementedError()

    def get_dev_examples(self, data_dir, **args):
        raise NotImplementedError()

    def get_test_examples(self, data_dir, **args):
        raise NotImplementedError()


class Woz2Processor(DataProcessor):
    def get_train_examples(self, data_dir, args):
        return dataset_woz2.create_examples(os.path.join(data_dir, 'woz_train_en.json'),
                                            'train', self.slot_list, self.domain_list, self.occur_list, self.label_maps, **args)

    def get_dev_examples(self, data_dir, args):
        return dataset_woz2.create_examples(os.path.join(data_dir, 'woz_validate_en.json'),
                                            'dev', self.slot_list, self.domain_list, self.occur_list, self.label_maps, **args)

    def get_test_examples(self, data_dir, args):
        return dataset_woz2.create_examples(os.path.join(data_dir, 'woz_test_en.json'),
                                            'test', self.slot_list, self.domain_list, self.occur_list, self.label_maps, **args)


class Multiwoz21Processor(DataProcessor):
    def get_train_examples(self, data_dir, args):
        return dataset_multiwoz21.create_examples(os.path.join(data_dir, 'train_dials.json'),
                                                  os.path.join(data_dir, 'dialogue_acts.json'),
                                                  'train', self.slot_list, self.domain_list, self.occur_list, self.label_maps, **args)

    def get_dev_examples(self, data_dir, args):
        return dataset_multiwoz21.create_examples(os.path.join(data_dir, 'val_dials.json'),
                                                  os.path.join(data_dir, 'dialogue_acts.json'),
                                                  'dev', self.slot_list, self.domain_list, self.occur_list, self.label_maps, **args)

    def get_test_examples(self, data_dir, args):
        return dataset_multiwoz21.create_examples(os.path.join(data_dir, 'test_dials.json'),
                                                  os.path.join(data_dir, 'dialogue_acts.json'),
                                                  'test', self.slot_list, self.domain_list, self.occur_list, self.label_maps, **args)


class SimProcessor(DataProcessor):
    def get_train_examples(self, data_dir, args):
        return dataset_sim.create_examples(os.path.join(data_dir, 'train.json'),
                                           'train', self.slot_list, self.domain_list, self.occur_list, **args)

    def get_dev_examples(self, data_dir, args):
        return dataset_sim.create_examples(os.path.join(data_dir, 'dev.json'),
                                           'dev', self.slot_list, self.domain_list, self.occur_list, **args)

    def get_test_examples(self, data_dir, args):
        return dataset_sim.create_examples(os.path.join(data_dir, 'test.json'),
                                           'test', self.slot_list, self.domain_list, self.occur_list, **args)


PROCESSORS = {"woz2": Woz2Processor,
              "sim-m": SimProcessor,
              "sim-r": SimProcessor,
              "multiwoz21": Multiwoz21Processor}
