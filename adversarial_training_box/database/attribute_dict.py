class AttributeDict(dict):
   __getattr__ = dict.__getitem__