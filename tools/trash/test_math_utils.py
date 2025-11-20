import isaaclab.utils.math as math_utils
import torch
import pdb
from scipy.spatial.transform import Rotation as R
# pdb.set_trace()
# rot = torch.eye(3)
# rot *= 0
# rot[0,2]=1
# rot[1,1]=1
# rot[2,0]=1
# math_utils.quat_from_matrix(rot)
# quaternion = R.from_matrix(rot.numpy()).as_quat()
import pdb

pdb.set_trace()
cur_quat = torch.tensor([[0.0, 0.77, 0.0, 0.77]])
inv_quat = torch.as_tensor([[0.0, 1.0, 0.0, 0.0]])
delta_quat = math_utils.quat_mul(cur_quat, math_utils.quat_inv(inv_quat))
math_utils.euler_xyz_from_quat(cur_quat)
target_quat = math_utils.quat_mul(delta_quat, inv_quat)
