__author__ = 'Sergey Matyunin'

import copy

class IntermediateSaver:
    def __init__(self, list_names_to_save):
        self._list_names_to_save = list_names_to_save
        self.data = []
    def save(self, obj):
        self.data.append({i:copy.copy(getattr(obj, i, None)) for i in self._list_names_to_save})