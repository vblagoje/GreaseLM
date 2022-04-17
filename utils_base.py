from collections import UserDict
from typing import Optional, Dict, Any, Union

import torch


class KGEncoding(UserDict):

    def __init__(self, data: Optional[Dict[str, Any]] = None):
        super().__init__(data)

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def _to_device(self, obj, device):
        if isinstance(obj, (tuple, list)):
            return [self._to_device(item, device) for item in obj]
        else:
            return obj.to(device)

    def to(self, device: Union[str, "torch.device"]) -> "BatchEncoding":
        """
        Send all values to device by calling `v.to(device)` (PyTorch only).
        Args:
            device (`str` or `torch.device`): The device to put the tensors on.
        Returns:
            [`KGEncoding`]: The same instance after modification.
        """

        # This check catches things like APEX blindly calling "to" on all inputs to a module
        # Otherwise it passes the casts down and casts the LongTensor containing the token idxs
        # into a HalfTensor
        if isinstance(device, str) or isinstance(device, torch.device) or isinstance(device, int):
            self.data = {k: self._to_device(obj=v, device=device) for k, v in self.data.items()}
        else:
            raise TypeError(f"device must be a str, torch.device, or int, got {type(device)}")
        return self
