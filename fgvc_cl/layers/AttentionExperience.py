from collections import namedtuple, deque
import random
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from functools import reduce

"""
Modified from 
https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch/blob/master/utilities/data_structures/Replay_Buffer.py
by Fangrui Liu (fangrui.liu@ubc.ca)
University of British Columbia Okanagan
"""




class AttentionExperience():
    """Replay buffer to store past experiences that the agent can then use for training data"""

    def __init__(self, classes, feature_channels=2048, buffer_size=5, avg_momentum=0.9, seed=None, priority=False):
        self.classes = classes
        self.buffer_size = buffer_size
        self.feature_channels = feature_channels
        #   TODO: Need to remove this to a prioritized list
        self.memory = [[] for i in range(classes)]
        self.experience = namedtuple("AttentionExperience",
                                     field_names=["label", "augmented", "distance", "age"])
        self.priority = priority
        if self.priority:
            self.avg_momentum = avg_momentum
            self.average_feature = nn.Parameter(torch.zeros(classes, feature_channels), requires_grad=False)
            self.criterion = nn.MSELoss()

        self.pickable_classes = np.arange(classes)
        self.seed = random.seed(seed)
        self.type_map = {
            "label": lambda x: x.long(),
            "augmented": lambda x: x.float(),
            "distance": lambda x: x.float(),
        }


    def add_experience(self, labels, augmenteds, feature_aug_mean):
        """Adds experience(s) into the replay buffer"""
        feature_aug_mean = torch.as_tensor(feature_aug_mean)
        for idx, label in enumerate(labels):
            if self.priority:
                with torch.no_grad():
                    s_feature_aug_mean = F.adaptive_avg_pool2d(
                        torch.as_tensor(feature_aug_mean[idx]).unsqueeze(0), 1).squeeze()
                    distance = self.criterion(s_feature_aug_mean, self.average_feature[label])
                    self.average_feature = self.avg_momentum * self.average_feature[label] + \
                                           (1.0 - self.avg_momentum) * s_feature_aug_mean
            else:
                distance = 0
            experience = self.experience(label, augmenteds[idx], distance, 0)
            self.memory[label].append(experience)

    def sample(self, avoid_class=None, num_experiences=20,
               separate_out_data_types=True, pick_random_class=False):
        """Draws a random sample of experience from the replay buffer"""
        experiences = self.pick_experiences(num_experiences,
                                            avoid_class=avoid_class,
                                            pick_random=pick_random_class)
        if separate_out_data_types:
            label, augmented = self.separate_out_data_types(experiences)
            return label, augmented
        else:
            return experiences

    def separate_out_data_types(self, experiences):
        """Puts the sampled experience into the correct format for a PyTorch neural network"""
        out = {
            "label": None,
            "augmented": None,
        }
        if len(experiences) > 0:
            for n in out.keys():
                if n not in ['augmented']:
                    out[n] = self.type_map[n](torch.from_numpy(
                        np.vstack([getattr(e, n) for e in experiences if e is not None])))
                elif n == 'augmented':
                    out[n] = self.type_map[n](torch.from_numpy(
                        np.stack([getattr(e, n) for e in experiences if e is not None], axis=0)))
        return out["label"], out["augmented"]

    def pick_experiences(self, num_experiences, avoid_class=None, pick_random=False):
        cursor = [0 for n in range(len(self.memory))]
        assert type(avoid_class) in [int, list, torch.Tensor] or avoid_class is None
        if type(avoid_class) is int:
            avoid_class = [avoid_class]
        l = []
        num_samples_each_classes = num_experiences // (self.classes if avoid_class is None else self.classes - 1)
        remainder = num_experiences
        if avoid_class is not None:
            _pick_class = np.delete(self.pickable_classes, avoid_class, axis=0)
        else:
            _pick_class = self.pickable_classes
        if num_samples_each_classes >= 1 and not pick_random:
            for n in _pick_class.tolist():
                l.extend(random.sample(self.memory[n], k=num_samples_each_classes))
            remainder -= num_samples_each_classes * self.classes

        picked_classes = _pick_class[np.random.randint(len(_pick_class), size=remainder).tolist()]
        for x in picked_classes:
            if len(self.memory[x]) > 0:
                if pick_random:
                    l.append(random.sample(self.memory[x], 1)[0])
                else:
                    if cursor[x] < len(self.memory[x]):
                        l.append(self.memory[x][cursor[x]])
                        cursor[x] += 1
        return l


    def sort(self):
        #   TODO (fangrui): Implement sorting stuff
        #   Criterion: contrastive distance
        #   delete the remaining
        for n in range(len(self.memory)):
            #   sort
            self.memory[n].sort(key=lambda x: x.distance + 0.001 * x.age)
            #   cut the list
            remainder = max(len(self.memory[n]) - self.buffer_size, 0)
            del self.memory[n][:remainder]
            #   aging
            for m in range(len(self.memory[n])):
                self.memory[n][m] = self.memory[n][m]._replace(age=self.memory[n][m].age + 1)


    def __len__(self):
        return list(reduce(lambda x, y: x+len(y) if type(x) is not list else len(x) + len(y), self.memory))


if __name__ == '__main__':
    from fgvc_cl.utils.Profiler import track, statistic_track
    from fgvc_cl.data.Transforms import norm, denorm
    classes = 200
    buff_size = 8
    feature_channels = 2048
    att_exp = AttentionExperience(classes, feature_channels, buffer_size=buff_size)


    def build_exp(exp):
        #   Build Experience
        for n in range(classes):
            exp.add_experience([n]*buff_size,
                               np.random.rand(buff_size, 3, 448, 448),
                               np.random.rand(buff_size, feature_channels, 14, 14))
        return exp


    wrapped_build = track(build_exp)
    wrapped_get = statistic_track(att_exp.sample)
    wrapped_sort = statistic_track(att_exp.sort)
    att_exp = wrapped_build(att_exp)
    label, augmented = wrapped_get(num_experiences=20, avoid_class=list(range(198)))
    wrapped_sort()
    print(label)

