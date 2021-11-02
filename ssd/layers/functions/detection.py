import torch
from torch.autograd import Function
from ..box_utils import decode, nms


class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, cfg, num_classes, top_k, conf_thresh, gpus):
        self.num_classes = num_classes
        self.top_k = top_k
        self.variance = cfg['variance']
        self.conf_thresh = min(conf_thresh)
        self.gpus = gpus

    def forward(self, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        output = torch.zeros(num, self.num_classes,
                             self.top_k, 5, device=self.gpus)
        conf_preds = conf_data.view(num, num_priors,
                                    self.num_classes).transpose(2, 1)

        # Decode predictions into bboxes.
        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()
            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.dim() == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                _, sort_scores = scores.sort(descending=True)
                topk_scores = scores[sort_scores][:self.top_k]
                topk_boxes = boxes[sort_scores][:self.top_k]
                count = topk_scores.size()[0]
                output[i, cl, :count] = \
                    torch.cat((topk_scores.unsqueeze(1),
                               topk_boxes), 1)
        return output
