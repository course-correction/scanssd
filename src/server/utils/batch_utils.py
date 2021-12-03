import torch


class Batch:

    def __init__(self):
        self.windows = None
        self.h = None
        self.w = None
        self.pad_h = None
        self.pad_w = None
        self.img_id = None

    def add_to_batch(self, windows: torch.Tensor, h: int, w: int, pad_h: int, pad_w: int, img_id: int):
        print("adding to batch")
        if self.windows is None:
            self.windows = windows
        else:
            self.windows = torch.cat((self.windows, windows))

        if self.h is None:
            self.h = torch.tensor(h).unsqueeze(0)
        else:
            self.h = torch.cat((self.h, torch.tensor(h).unsqueeze(0)))

        if self.w is None:
            self.w = torch.tensor(w).unsqueeze(0)
        else:
            self.w = torch.cat((self.w, torch.tensor(w).unsqueeze(0)))

        if self.pad_h is None:
            self.pad_h = torch.tensor(pad_h).unsqueeze(0)
        else:
            self.pad_h = torch.cat((self.pad_h, torch.tensor(pad_h).unsqueeze(0)))

        if self.pad_w is None:
            self.pad_w = torch.tensor(pad_w).unsqueeze(0)
        else:
            self.pad_w = torch.cat((self.pad_w, torch.tensor(pad_w).unsqueeze(0)))

        if self.img_id is None:
            self.img_id = torch.tensor(img_id).unsqueeze(0)
        else:
            self.img_id = torch.cat((self.img_id, torch.tensor(img_id).unsqueeze(0)))




