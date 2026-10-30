# Copyright (c) CryoNet Team, and its affiliates. All Rights Reserved

import numpy as np
from enum import Enum, unique
from typing import Iterator, List, Tuple, Union

import torch, math


_RawBoxType = Union[List[float], Tuple[float, ...], torch.Tensor, np.ndarray]

def cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)

@unique
class BoxMode(Enum):
    """
    Enum of different ways to represent a box.

    Attributes:

        XYXY_ABS: (x0, y0, z0, x1, y1, z1) in absolute floating points coordinates.
            The coordinates in range [0, width or height or length].
        XYWH_ABS: (x0, y0, z0, w, h, l) in absolute floating points coordinates.
        XYXY_REL: (x0, y0, z0, x1, y1, z1) in range [0, 1]. They are relative to the size of the image.
        XYWH_REL: (x0, y0, z0, w, h, l) in range [0, 1]. They are relative to the size of the image.
    """

    XYZXYZ_ABS = 0
    XYZWHL_ABS = 1
    XYZXYZ_REL = 2
    XYZWHL_REL = 3

    @staticmethod
    def convert(box: _RawBoxType, from_mode: "BoxMode", to_mode: "BoxMode") -> _RawBoxType:
        """
        Args:
            box: can be a 6-tuple, 6-list or a Nx6 array/tensor.
            from_mode, to_mode (BoxMode)

        Returns:
            The converted box of the same type.
        """
        if from_mode == to_mode:
            return box

        original_type = type(box)
        single_box = isinstance(box, (list, tuple))
        if single_box:
            arr = np.array(box)
            assert arr.shape == (
                6,
            ), "BoxMode.convert takes either a 6-tuple/list or a Nx6 array/tensor"
        else:
            arr = box

        assert to_mode.value < 2 and from_mode.value < 2, "Relative mode not yet supported!"

        original_shape = arr.shape
        arr = arr.reshape(-1, 6)
        if to_mode == BoxMode.XYZXYZ_ABS and from_mode == BoxMode.XYZWHL_ABS:
            arr[:, 3] += arr[:, 0]
            arr[:, 4] += arr[:, 1]
            arr[:, 5] += arr[:, 2]
        elif from_mode == BoxMode.XYZXYZ_ABS and to_mode == BoxMode.XYZWHL_ABS:
            arr[:, 3] -= arr[:, 0]
            arr[:, 4] -= arr[:, 1]
            arr[:, 5] -= arr[:, 2]
        else:
            raise RuntimeError("Cannot be here!")
        if single_box:
            return original_type(arr.flatten())
        return arr.reshape(*original_shape)



class Boxes:
    """
    This structure stores a list of boxes as a Nx6 torch.Tensor.
    It supports some common methods about boxes
    (`area`, `clip`, `nonempty`, etc),
    and also behaves like a Tensor
    (support indexing, `to(device)`, `.device`, and iteration over all boxes)

    Attributes:
        tensor: float matrix of Nx6.
    """

    BoxSizeType = Union[List[int], Tuple[int, int]]

    def __init__(self, 
                 tensor: _RawBoxType, 
                 class_idx : int = 0, 
                 path_idx : int = 0, 
                 path_len : int = 0, 
                 voxel_size : float = 1,
                 pdb_path : bool = False):
        """
        Args:
            tensor (Tensor[float]): a Nx6 matrix.  Each row is (x1, y1, z1, x2, y2, z2).
        """
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
        device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        # import pdb;pdb.set_trace()
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        tensor[:,:6] = tensor[:,:6] * voxel_size

        if tensor.numel() == 0:
            tensor = torch.zeros(0, 6, dtype=torch.float32, device=device)
        assert tensor.dim() == 2 and tensor.size(-1) >= 6 and tensor.size(0) > 0, tensor.size()

        self.tensor   = tensor[:,:]
        # self.tensor  = tensor[:,:6]
        self.classes  = torch.zeros(tensor.size(0), dtype=torch.int32, device=device)
        # self.path     = torch.zeros(tensor.size(0), dtype=torch.int32, device=device) - 1 
        self.scores   = torch.zeros(tensor.size(0), dtype=torch.int32, device=device)
        self.poses    = torch.zeros(tensor.size(0), 14, 3, dtype=torch.float32, device=device)
        self.ters     = torch.ones(tensor.size(0), dtype=torch.int32, device=device)
        self.is_fill  = torch.zeros(tensor.size(0), dtype=torch.int32, device=device)
        
        if class_idx == 6 or tensor.size(-1)>= 7:
            self.classes = tensor[:,6].int()

        if path_idx == 7:
            self.path = (tensor[:,7] - tensor[:,7].min()).int()
        elif path_len > 0:
            self.path = torch.arange(path_len)
        else:
            self.path = torch.arange(tensor.size(0))
        if pdb_path:
            self.pdb_path = (tensor[:, 7].int())

        if tensor.size(-1) >= 8:
            self.chain = tensor[:, 8]
        else:
            self.chain = torch.zeros(tensor.size(0))

        if tensor.size(-1) >= 10:
            self.sses = tensor[:, 9].int()
        else:
            self.sses = torch.zeros(tensor.size(0), dtype=torch.int32, device=device)
        if tensor.size(-1) >= 11:
            self.scores = tensor[:, 10]
        else:
            self.scores = torch.zeros(tensor.size(0), dtype=torch.float32, device=device)

        

    def clone(self) -> "Boxes":
        """
        Clone the Boxes.

        Returns:
            Boxes
        """
        cl = Boxes(self.tensor.clone())
        cl.set_classes(self.classes.clone())
        cl.set_scores(self.scores.clone())
        cl.set_poses(self.poses.clone())
        cl.set_path(self.path.clone())
        cl.set_chain(self.chain.clone())
        cl.set_sses(self.sses.clone())
        cl.set_scores(self.scores.clone())
        return cl

    def to(self, device: str) -> "Boxes":
        return Boxes(self.tensor.to(device))

    def set_classes(self, classes) -> torch.Tensor:
        """
        set the classes of all the boxes.

        Returns:
            torch.Tensor: a vector with classes of each box.
        """
        assert self.tensor.size(0) == len(classes)
        self.classes = torch.as_tensor(classes, dtype=torch.int32, device=self.device)

    def get_classes(self) -> torch.Tensor:
        """
        get the classes of all the boxes.

        Returns:
            torch.Tensor: a vector with classes of each box.
        """
        return self.classes.int()

    def set_path(self, path) -> torch.Tensor:
        """
        set the path of all the boxes.

        Returns:
            torch.Tensor: a vector with path of each box.
        """
        assert len(self.path) == len(path), "path length ({}) is not matched with the target path length ({})".format( len(path), len(self.path))
        self.path = torch.as_tensor(path, dtype=torch.int32, device=self.device)
    
    def get_path(self) -> torch.Tensor:
        """
        get the path of all the boxes.

        Returns:
            torch.Tensor: a vector with path of each box.
        """
        return self.path

    def get_pdb_path(self) -> torch.Tensor:
        """
        get the path of all the boxes.

        Returns:
            torch.Tensor: a vector with path of each box.
        """
        return self.pdb_path

    def set_chain(self, chain) -> torch.Tensor:
        """
        set the chain of all the boxes.

        Returns:
            torch.Tensor: a vector with chain of each box.
        """
        assert self.tensor.size(0) == len(chain)
        self.chain = torch.as_tensor(chain, dtype=torch.int32, device=self.device)

    def get_chain(self) -> torch.Tensor:
        """
        get the chain of all the boxes.

        Returns:
            torch.Tensor: a vector with chain of each box.
        """
        return self.chain.int()

    def set_sses(self, sses) -> torch.Tensor:
        """
        set the sses of all the boxes.

        Returns:
            torch.Tensor: a vector with sses of each box.
        """
        assert self.tensor.size(0) == len(sses)
        self.sses = torch.as_tensor(sses, dtype=torch.int32, device=self.device)

    def get_sses(self) -> torch.Tensor:
        """
        get the sses of all the boxes.

        Returns:
            torch.Tensor: a vector with sses of each box.
        """
        return self.sses.int()
    
    def set_scores(self, scores) -> torch.Tensor:
        """
        set the scores of all the boxes.

        Returns:
            torch.Tensor: a vector with scores of each box.
        """
        if isinstance(scores, np.ndarray):
            scores = torch.from_numpy(scores)
        assert self.tensor.size(0) == len(scores)
        self.scores = torch.as_tensor(scores, dtype=torch.float32, device=self.device)

    def get_scores(self) -> torch.Tensor:
        """
        get the scores of all the boxes.

        Returns:
            torch.Tensor: a vector with scores of each box.
        """
        return self.scores

    def set_ters(self, ters) -> torch.Tensor:
        """
        set the scores of all the boxes.

        Returns:
            torch.Tensor: a vector with ters of each box.
        """
        if isinstance(ters, np.ndarray):
            ters = torch.from_numpy(ters).int()
        assert self.tensor.size(0) == len(ters)
        self.ters = torch.as_tensor(ters, dtype=torch.float32, device=self.device)

    def get_ters(self) -> torch.Tensor:
        """
        get the scores of all the boxes.

        Returns:
            torch.Tensor: a vector with scores of each box.
        """
        return self.ters
    
    def set_poses(self, poses, voxel_size=1) -> torch.Tensor:
        """
        set the backbone pose of all the boxes. (n,14,3)

        Returns:
            torch.Tensor: a vector with poses of each box.
        """
        if isinstance(poses, np.ndarray):
            poses = torch.from_numpy(poses)
        assert self.tensor.size(0) == len(poses), "box: {}, coords: {}".format(self.tensor.size(0), len(poses))
        # assert poses.size(1) == 4
        assert poses.size(2) == 3
        self.poses = torch.as_tensor(poses * voxel_size, dtype=torch.float32, device=self.device)

    def get_poses(self, full=False) -> torch.Tensor:
        """
        get the poses of all the boxes.

        Returns:
            torch.Tensor: a vector with poses of each box.
        """
        return self.poses

    # def set_fposes(self, fposes) -> torch.Tensor:
    #     """
    #     set the full atom pose of all the boxes.

    #     Returns:
    #         torch.Tensor: a vector with full atom pose of each box.
    #     """
    #     assert self.tensor.size(0) == len(fposes)
    #     self.fposes = torch.as_tensor(fposes, dtype=torch.float32, device=self.device)

    # def get_fposes(self) -> torch.Tensor:
    #     """
    #     get the full atom pose of all the boxes.

    #     Returns:
    #         torch.Tensor: a vector with full atom pose of each box.
    #     """
    #     return self.fposes

    # def set_ca(self, ca) -> torch.Tensor:
    #     """
    #     set the Ca of all the boxes.

    #     Returns:
    #         torch.Tensor: a vector with Ca of each box.
    #     """
    #     assert self.tensor.size(0) == len(ca)
    #     assert ca.size(1) == 3
    #     self.ca = torch.as_tensor(ca, dtype=torch.float32, device=self.device)

    def get_centers(self) -> torch.Tensor:
        """
        Returns:
            The box centers in a Nx3 array of (x, y, z).
        """
        return (self.tensor[:, :3] + self.tensor[:, 3:6] + 1 ) / 2
    
    def get_angles(self, ca=True) -> torch.Tensor:
        """
        Returns:
            N-2 angles.
        """
        # bond length, angle, dihedral (numpy version) -------
        # def angle_np(a,b,c):
        #     ba = a - b
        #     bc = c - b
        #     cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        #     angle = np.arccos(cosine_angle)
        #     return np.degrees(angle)
        # cas = self.get_ca().numpy()
        # angles = torch.zeros(len(cas)-2)
        # # import pdb; pdb.set_trace()
        # for i in range(0, len(cas)-2):
        #     print(i,"np", angle_np(cas[i], cas[i+1], cas[i+2]))
        # print(angle_pt())

        if ca:
            cas = self.get_ca()
        else:
            cas = self.get_centers()
        angles = torch.ones(len(cas)) * 104.2654
        l = len(cas)
        ba = cas[0:l-2] - cas[1:l-1]
        bc = cas[2:l]   - cas[1:l-1]
        inner_product = (ba * bc).sum(dim=1)
        a_norm = ba.pow(2).sum(dim=1).pow(0.5) # actually, normal can be obtained from distance matrix
        b_norm = bc.pow(2).sum(dim=1).pow(0.5)
        cos = inner_product / (a_norm * b_norm)
        angles_p = torch.acos(torch.clamp(cos, -1, 1))* 180/math.pi
        angles[1:l-1] = angles_p
        return angles

    def get_angle(self, i, j, k, ca=True, return_norm=False) -> torch.Tensor:
        if ca:
            cas = self.get_ca()
        else:
            cas = self.get_centers()
        a,b,c = cas[i:i+1], cas[j:j+1], cas[k:k+1]
        a,b,c = cas[i], cas[j], cas[k]
        ba = a - b
        bc = c - b
        inner_product = (ba * bc).sum()
        a_norm = ba.pow(2).sum().pow(0.5)
        b_norm = bc.pow(2).sum().pow(0.5)
        # print(a_norm,b_norm)
        cos = inner_product / (a_norm * b_norm)
        angle = torch.acos(torch.clamp(cos, -1, 1))* 180/math.pi
        if return_norm:
            return angle, a_norm, b_norm
        else:
            return angle
    
    def set_fills(self, is_fill) -> torch.Tensor:
        if isinstance(is_fill, np.ndarray):
            is_fill = torch.from_numpy(is_fill)
        assert self.tensor.size(0) == len(is_fill)
        self.is_fill = torch.as_tensor(is_fill, dtype=torch.float32, device=self.device)

    def get_fills(self) -> torch.Tensor:
        """
        get the fills of all the boxes.
        Returns:
            torch.Tensor: a vector with scores of each box.
        """
        return self.is_fill
    
    def set_res_nums(self, res_id_num_list) -> torch.Tensor:
        if isinstance(res_id_num_list, np.ndarray):
            is_fill = torch.from_numpy(res_id_num_list)
        assert self.tensor.size(0) == len(res_id_num_list)
        self.res_nums = torch.as_tensor(res_id_num_list, dtype=torch.int32, device=self.device)

    def get_res_nums(self) -> torch.Tensor:
        """
        get the res_nums of all the boxes.
        Returns:
            torch.Tensor: a vector with scores of each box.
        """
        return self.res_nums

    def set_chain_ids(self, chain_ids) -> torch.Tensor:
        if isinstance(chain_ids, np.ndarray):
            chain_ids = torch.from_numpy(chain_ids)
        assert self.tensor.size(0) == len(chain_ids)
        self.chain_ids = torch.as_tensor(chain_ids, dtype=torch.int32, device=self.device)

    def get_chain_ids(self) -> torch.Tensor:
        """
        get the chain_ids of all the boxes.
        Returns:
            torch.Tensor: a vector with scores of each box.
        """
        return self.chain_ids

    def get_whole_path_list(self):
        res_number=len(self.classes)
        chain_number=self.chain_ids[-1]+1
        whole_path_list=[[] for i in range(chain_number)]
        for res_index in range(res_number):
            chain_pos=self.res_nums[res_index]
            chain_id=self.chain_ids[res_index]
            while(len(whole_path_list[chain_id])<=chain_pos):
                whole_path_list[chain_id].append(-1)
            whole_path_list[chain_id][chain_pos]=res_index
        return whole_path_list

    
    def get_start(self) -> torch.Tensor:
        """
        Returns:
            The box start in a Nx3 array of (x, y, z).
        """
        return self.tensor[:, :3] #+ self.tensor[:, 3:6] + 1 ) / 2

    def get_ca(self) -> torch.Tensor:
        """
        get the Ca of all the boxes.

        Returns:
            torch.Tensor: a vector with Ca of each box.
        """
        # assert self.ca.min() < self.ca.max(), "Ca is empty!"
        return self.get_poses()[:,1,:]

    def volume(self) -> torch.Tensor:
        """
        Computes the volume of all the boxes.

        Returns:
            torch.Tensor: a vector with volume of each box.
        """
        box = self.tensor
        volume = (box[:, 3] - box[:, 0] + 1 ) * (box[:, 4] - box[:, 1] + 1 ) * (box[:, 5] - box[:, 2] + 1 )
        return volume

    def edge(self) -> torch.Tensor:
        """
        Computes the edge of all the boxes.

        Returns:
            torch.Tensor: a vector with areas of each box.
        """
        box = self.tensor
        return self.width, self.height, self.length

    def width(self) -> torch.Tensor:
        """
        Computes the width of all the boxes.

        Returns:
            torch.Tensor: a vector with width of each box.
        """
        box = self.tensor
        return box[:, 3] - box[:, 0] + 1

    def height(self) -> torch.Tensor:
        """
        Computes the height of all the boxes.

        Returns:
            torch.Tensor: a vector with height of each box.
        """
        box = self.tensor
        return box[:, 4] - box[:, 1] + 1

    def length(self) -> torch.Tensor:
        """
        Computes the edge of all the boxes.

        Returns:
            torch.Tensor: a vector with areas of each box.
        """
        box = self.tensor
        return box[:, 5] - box[:, 2] + 1
    
    def region(self) -> torch.Tensor:
        """
        Computes the region of all the boxes.

        Returns:
            torch.Tensor: a vector with areas of each box.
        """
        box = self.tensor
        return min(box[:, 0].min(), box[:, 3].min()),  \
               min(box[:, 1].min(), box[:, 4].min()),  \
               min(box[:, 2].min(), box[:, 5].min()),  \
               max(box[:, 0].max(), box[:, 3].max()),  \
               max(box[:, 1].max(), box[:, 4].max()),  \
               max(box[:, 2].max(), box[:, 5].max())
                
        

    def clip(self, box_size: BoxSizeType, inplace=True) -> None:
        """
        Clip (in place) the boxes by limiting x coordinates to the range [0, width]
        and y coordinates to the range [0, height] and z coordinates to the range [0, length].

        Args:
            box_size (width, height, length): The clipping box's size.
        """
        assert torch.isfinite(self.tensor).all()
        w, h, l = box_size
        if inplace:
            self.tensor[:, 0].clamp_(min=0, max=w)
            self.tensor[:, 1].clamp_(min=0, max=h)
            self.tensor[:, 2].clamp_(min=0, max=l)
            self.tensor[:, 3].clamp_(min=0, max=w)
            self.tensor[:, 4].clamp_(min=0, max=h)
            self.tensor[:, 5].clamp_(min=0, max=l)
            return self
        else:
            boxes = self.tensor.clone()
            boxes[:, 0].clamp_(min=0, max=w)
            boxes[:, 1].clamp_(min=0, max=h)
            boxes[:, 2].clamp_(min=0, max=l)
            boxes[:, 3].clamp_(min=0, max=w)
            boxes[:, 4].clamp_(min=0, max=h)
            boxes[:, 5].clamp_(min=0, max=l)
            boxes_tmp = self.clone()
            boxes_tmp.tensor = boxes
            return boxes_tmp

    

    def shift(self, r) -> torch.Tensor:
        boxes = self.tensor.clone()
        boxes[:,0] -= r[0]
        boxes[:,1] -= r[1]
        boxes[:,2] -= r[2]
        boxes[:,3] -= r[0]
        boxes[:,4] -= r[1]
        boxes[:,5] -= r[2]
        return boxes

    def shift_(self, r) -> torch.Tensor:
        self.tensor[:,0] -= r[0]
        self.tensor[:,1] -= r[1]
        self.tensor[:,2] -= r[2]
        self.tensor[:,3] -= r[0]
        self.tensor[:,4] -= r[1]
        self.tensor[:,5] -= r[2]
        return self.tensor

    def shift_plus(self, r) -> torch.Tensor:
        boxes = self.tensor.clone()
        boxes[:,0] += r[0]
        boxes[:,1] += r[1]
        boxes[:,2] += r[2]
        boxes[:,3] += r[0]
        boxes[:,4] += r[1]
        boxes[:,5] += r[2]
        boxes_tmp = self.clone()
        boxes_tmp.tensor = boxes
        return boxes_tmp
    
    def shift_plus_(self, r) -> torch.Tensor:
        self.tensor[:,0] += r[0]
        self.tensor[:,1] += r[1]
        self.tensor[:,2] += r[2]
        self.tensor[:,3] += r[0]
        self.tensor[:,4] += r[1]
        self.tensor[:,5] += r[2]
    
    

    def nonempty(self, threshold: int = 3) -> torch.Tensor:
        """
        Find boxes that are non-empty.
        A box is considered empty, if either of its side is no larger than threshold.

        Returns:
            Tensor:
                a binary vector which represents whether each box is empty
                (False) or non-empty (True).
        """
        # box = self.tensor
        # widths  = box[:, 3] - box[:, 0] 
        # heights = box[:, 4] - box[:, 1]
        # lengths = box[:, 5] - box[:, 2]
        keep = (self.width() > threshold) & (self.height() > threshold) & (self.length() > threshold)
        return keep

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Boxes":
        """
        Returns:
            Boxes: Create a new :class:`Boxes` by indexing.

        The following usage are allowed:
        1. `new_boxes = boxes[3]`: return a `Boxes` which contains only one box.
        2. `new_boxes = boxes[2:10]`: return a slice of boxes.
        3. `new_boxes = boxes[vector]`, where vector is a torch.BoolTensor
           with `length = len(boxes)`. Nonzero elements in the vector will be selected.

        Note that the returned Boxes might share storage with this Boxes,
        subject to Pytorch's indexing semantics.
        self.classes  = torch.zeros(tensor.size(0), dtype=torch.int32, device=device)
        self.path    = torch.zeros(tensor.size(0), dtype=torch.int32, device=device) - 1 
        self.scores   = torch.zeros(tensor.size(0), dtype=torch.int32, device=device)
        self.poses    = torch.zeros(tensor.size(0), 4, 3, dtype=torch.float32, device=device)
        self.fposes   = []
        """
        if isinstance(item, int):
            ibox = Boxes(self.tensor[item].view(1, -1))
            ibox.set_classes(self.get_classes()[item].view(1, -1))
            ibox.set_scores(self.get_scores()[item].view(1, -1))
            ibox.set_poses(self.get_poses()[item].view(1, -1, 3))
            # ibox.set_poses(self.get_poses()[item].view(1, -1, 3))
            ibox.set_path([0])
            return ibox
        b = self.tensor[item]
        assert b.dim() == 2, "Indexing on Boxes with {} failed to return a matrix!".format(item)
        ibox = Boxes(b)
        ibox.set_classes(self.get_classes()[item])
        ibox.set_scores(self.get_scores()[item])
        ibox.set_poses(self.get_poses()[item])
        # ibox.set_poses(self.get_poses()[item].view(1, -1, 3))
        # ibox.set_path(list(range(len(item))))
        return ibox


    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __repr__(self) -> str:
        return "Boxes(" + str(self.tensor) + ")"

    def inside_box(self, box_size: BoxSizeType, boundary_threshold: int = 0) -> torch.Tensor:
        """
        Args:
            box_size (width, height, length): Size of the reference box.
            boundary_threshold (int): Boxes that extend beyond the reference box
                boundary by more than boundary_threshold are considered "outside".

        Returns:
            a binary vector, indicating whether each box is inside the reference box.
        """
        width, height, length = box_size
        inds_inside = (
            (self.tensor[..., 0] >= -boundary_threshold)
            & (self.tensor[..., 1] >= -boundary_threshold)
            & (self.tensor[..., 2] >= -boundary_threshold)
            & (self.tensor[..., 3] < width + boundary_threshold)
            & (self.tensor[..., 4] < height + boundary_threshold)
            & (self.tensor[..., 5] < length + boundary_threshold)
        )
        return inds_inside


    def info(self):
        print("min: ", self.tensor[:,:6].min(dim=0))
        print("max: ", self.tensor[:,:6].max(dim=0))

    @staticmethod
    def cat(boxes_list: List["Boxes"]) -> "Boxes":
        """
        Concatenates a list of Boxes into a single Boxes

        Arguments:
            boxes_list (list[Boxes])

        Returns:
            Boxes: the concatenated Boxes
        """
        assert isinstance(boxes_list, (list, tuple))
        assert len(boxes_list) > 0
        assert all(isinstance(box, Boxes) for box in boxes_list)

        cat_boxes = type(boxes_list[0])(cat([b.tensor for b in boxes_list], dim=0))
        return cat_boxes

    @property
    def device(self) -> str:
        return self.tensor.device

    def __iter__(self) -> Iterator[torch.Tensor]:
        """
        Yield a box as a Tensor of shape (6,) at a time.
        """
        yield from self.tensor

def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    edist = lambda A, B: torch.sqrt(torch.clamp((A.pow(2).sum(1, keepdim = True) - \
        2 * torch.mm(A, B.t()) + B.pow(2).sum(1, keepdim = True).t()), min=0))
    return edist(x, y)


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def pairwise_iou(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    """
    Given two lists of boxes of size N and M,
    compute the IoU (intersection over union)
    between __all__ N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        Tensor: IoU, sized [N,M].
    """
    volume1 = boxes1.volume()
    volume2 = boxes2.volume()

    boxes1, boxes2 = boxes1.tensor[:,:6], boxes2.tensor[:,:6]

    lt = torch.max(boxes1[:, None, :3], boxes2[:, :3])  # [N,M,3]
    rb = torch.min(boxes1[:, None, 3:], boxes2[:, 3:])  # [N,M,3]

    whl = (rb + 1 - lt).clamp(min=0)  # [N,M,3]
    inter = whl[:, :, 0] * whl[:, :, 1] * whl[:, :, 2]  # [N,M]

    # handle empty boxes
    iou = torch.where(
        inter > 0,
        inter / (volume1[:, None] + volume2 - inter),
        torch.zeros(1, dtype=inter.dtype, device=inter.device),
    )
    return iou


def matched_boxlist_iou(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    """
    Compute pairwise intersection over union (IOU) of two sets of matched
    boxes. The box order must be (xmin, ymin, zmin, xmax, ymax, zmax).
    Similar to boxlist_iou, but computes only diagonal elements of the matrix
    Arguments:
        boxes1: (Boxes) bounding boxes, sized [N,6].
        boxes2: (Boxes) bounding boxes, sized [N,6].
    Returns:
        (tensor) iou, sized [N].
    """
    assert len(boxes1) == len(boxes2), (
        "boxlists should have the same"
        "number of entries, got {}, {}".format(len(boxes1), len(boxes2))
    )
    volume1 = boxes1.volume()
    volume2 = boxes2.volume()
    box1, box2 = boxes1.tensor[:,:6], boxes2.tensor[:,:6]
    lt = torch.max(box1[:, :3], box2[:, :3])  # [N,3]
    rb = torch.min(box1[:, 3:], box2[:, 3:])  # [N,3]
    whl = (rb + 1 - lt).clamp(min=0)  # [N,3]
    inter = whl[:, 0] * whl[:, 1] * whl[:, 2]  # [N]
    iou = inter / (volume1 + volume2 - inter)  # [N]
    return iou
