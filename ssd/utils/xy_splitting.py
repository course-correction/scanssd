import torch

# XY Cut to split all overlapping regions
class SplitDoc(object):
    def __init__(self, idx, boxes, sorted_ids, split_check=False):
        self.idx = idx
        self.data = boxes
        self.sorted_ids = sorted_ids
        self.split_check = split_check
        self.children = []

    def add_child(self, split):
        assert isinstance(split, SplitDoc)
        self.children.append(split)

    def all_nodes_check(self, node, leaves):
        if node.split_check:
            raise Exception('Got a node which has already been checked for splits!')
        self.make_splits(node)

        if len(node.children):
            for child in node.children:
                self.all_nodes_check(child, leaves)
        else:
            leaves.append(node)
        return leaves

    @staticmethod
    def make_splits(node):
        boxes = node.data
        sorted_y = node.sorted_ids[1]
        sorted_x = node.sorted_ids[0]
        level = node.idx
        split_pt = []

        # Sort boxes according to ymins
        if level % 2 == 0:
            split_dir_ids = sorted_y
            other_dir_ids = sorted_x
            compare = 1
        # Sort boxes according to xmins
        else:
            split_dir_ids = sorted_x
            other_dir_ids = sorted_y
            compare = 0


        # # Create the splits
        # idx = 1
        # curr_split = 0
        # can_split = True
        # while idx < boxes.size(0):
        #     for b in range(curr_split, idx):
        #         # If any previous xmax/ymax greater than the compared xmin/ymin,
        #         # cannot split.
        #         if boxes[b, compare+2] > boxes[idx, compare]:
        #             can_split = False
        #             break
        #     # If no previous boxes found overlapping create a split point
        #     # Start comparing the next box from the new split
        #     if can_split:
        #         split_pt.append(idx)
        #         curr_split = idx
        #         idx += 1
        #     # Found overlap from a box, compare the next box with previous ones
        #     else:
        #         idx += 1
        #
        #     can_split = True


        # min_pt = boxes[split_dir_ids[0], compare]
        max_pt = boxes[split_dir_ids[0], compare+2]
        split_ids = []
        curr_split = 0
        # First box will always be in split 0
        id2split = {split_dir_ids[0]: curr_split}

        box_id = 1
        for idx, box_id in enumerate(split_dir_ids[1:]):
            if boxes[box_id, compare] > max_pt:
                curr_split += 1
                split_ids.append(idx+1)
                # min_pt = boxes[box_id, compare]
                max_pt = boxes[box_id, compare+2]
                id2split[box_id] = curr_split
            else:
                if boxes[box_id, compare+2] > max_pt:
                    max_pt = boxes[box_id, compare+2]
                id2split[box_id] = curr_split

        # If only there are splits, search for the other dir order
        if split_ids:
            split_dir_grps = []
            other_dir_grps = [[] for num_splits in range(curr_split+1)]

            # Get split direction groups first
            start = 0
            for idx in split_ids:
                split_dir_grps.append(split_dir_ids[start:idx])
                start = idx
            split_dir_grps.append(split_dir_ids[start:])

            # Now get the other direction groups
            for oth_box_id in other_dir_ids:
                split_grp = id2split[oth_box_id]
                other_dir_grps[split_grp].append(oth_box_id)

            # assert len(split_dir_grps) == len(other_dir_grps), \
            #     'Num Split mismatch between X and Y direction'

            # Now make the child nodes
            for idx, grp in enumerate(split_dir_grps):
                # assert len(grp) == len(other_dir_grps[idx]), \
                #     'Num boxes in X and Y dir for split {} do not match. ' \
                #     'Sizes: {} and {}'.format(idx, len(grp), len(other_dir_grps[idx]))
                if compare == 0:
                    new_node = SplitDoc(level+1, boxes, [grp, other_dir_grps[idx]],
                                        split_check=False)
                else:
                    new_node = SplitDoc(level+1, boxes, [other_dir_grps[idx],grp],
                                        split_check=False)
                node.add_child(new_node)

            node.split_check = True



        # # Make the splits into nodes if splits found
        # if len(split_pt):
        #     start = 0
        #     for pt in split_pt:
        #         new_node = SplitDoc(level+1, [boxes[start:pt], scores[start:pt]],
        #                             split_check=False)
        #         node.add_child(new_node)
        #         start = pt
        #     # Add the remaining boxes to a new split
        #     new_node = SplitDoc(level+1, [boxes[start:], scores[start:]],
        #                         split_check=False)
        #     node.add_child(new_node)
        #
        # node.split_check = True


def create_splits(boxes, scores, topk):
    boxes = boxes.clone()
    scores = scores.clone()

    # Get topk
    _, idx = scores.sort(descending=True)
    topk_idx = idx[:topk]
    topk_boxes = boxes[topk_idx]
    topk_scores = scores[topk_idx]

    level = 0

    # Map Horizontal and vertical sorted boxes to main boxes
    ys = topk_boxes[:, 1]
    _, y_ind = torch.sort(ys)
    xs = topk_boxes[:, 0]
    _, x_ind = torch.sort(xs, 0)

    # Convert sort indexes to numpy for using as keys
    y_ind = y_ind.data.cpu().numpy()
    x_ind = x_ind.data.cpu().numpy()

    # Create root Node with all boxes
    root = SplitDoc(level, topk_boxes, [x_ind, y_ind], split_check=False)

    # Create XY Split with alternating horizontal and vertical cuts and
    # get the leaves
    leaves = []
    all_leaves = root.all_nodes_check(root, leaves)

    # Get all data from leaf nodes
    splits = []
    for leaf in all_leaves:
        ids = leaf.sorted_ids[0]
        splits.append([topk_boxes[ids], topk_scores[ids]])
    return splits
