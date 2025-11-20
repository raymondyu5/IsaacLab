import collections
from copy import copy

import numpy as np
import isaaclab.utils.math as math_utils

import torch


# Function to compute the diagonal distance (radius approximation for the object)
def get_object_radius(size):
    return torch.norm(size) / 2.0


# Function to check if two objects are colliding
def check_collision(pos1, size1, pos2, size2):
    radius1 = get_object_radius(size1)
    radius2 = get_object_radius(size2)

    # Calculate the distance between the two object positions
    distance = torch.norm(pos1 - pos2)

    # Check if the distance is less than the sum of their radii (collision)
    return distance < (radius1 + radius2)


# Function to sample random positions within the bin without collisions
def sample_positions(bin_size, object_sizes, max_attempts=1000):
    positions = []

    for size in object_sizes:
        attempts = 0
        while attempts < max_attempts:
            # Sample a random position within the bin (account for object size to prevent overflow)
            pos = torch.rand(3, device='cuda:0') * (bin_size - size) + size / 2

            # Check for collisions with previously placed objects
            collision = False
            for i, prev_pos in enumerate(positions):
                if check_collision(pos, size, prev_pos, object_sizes[i]):
                    collision = True
                    break

            # If no collision, save the position and move to the next object
            if not collision:
                positions.append(pos)
                break
            attempts += 1

        # If max attempts are reached without finding a valid position, raise an error
        if attempts == max_attempts:
            raise ValueError(
                f"Could not place object of size {size} without collision after {max_attempts} attempts."
            )

    return positions


class ObjectPositionSampler:
    """
    Base class of object placement sampler.

    Args:
        name (str): Name of this sampler.

        mujoco_objects (None or MujocoObject or list of MujocoObject): single model or list of MJCF object models

        ensure_object_boundary_in_range (bool): If True, will ensure that the object is enclosed within a given boundary
            (should be implemented by subclass)

        ensure_valid_placement (bool): If True, will check for correct (valid) object placements

        reference_pos (3-array): global (x,y,z) position relative to which sampling will occur

        z_offset (float): Add a small z-offset to placements. This is useful for fixed objects
            that do not move (i.e. no free joint) to place them above the table.
    """

    def __init__(
            self,
            name,
            mujoco_objects=None,
            ensure_object_boundary_in_range=True,
            ensure_valid_placement=True,
            reference_pos=(0, 0, 0),
            z_offset=0.0,
    ):
        # Setup attributes
        self.name = name
        if mujoco_objects is None:
            self.mujoco_objects = []
        else:
            # Shallow copy the list so we don't modify the inputted list but still keep the object references
            self.mujoco_objects = [mujoco_objects] if isinstance(
                mujoco_objects, MujocoObject) else copy(mujoco_objects)
        self.ensure_object_boundary_in_range = ensure_object_boundary_in_range
        self.ensure_valid_placement = ensure_valid_placement
        self.reference_pos = reference_pos
        self.z_offset = z_offset

    def add_objects(self, mujoco_objects):
        """
        Add additional objects to this sampler. Checks to make sure there's no identical objects already stored.

        Args:
            mujoco_objects (MujocoObject or list of MujocoObject): single model or list of MJCF object models
        """
        mujoco_objects = [mujoco_objects] if isinstance(
            mujoco_objects, MujocoObject) else mujoco_objects
        for obj in mujoco_objects:
            assert obj not in self.mujoco_objects, "Object '{}' already in sampler!".format(
                obj.name)
            self.mujoco_objects.append(obj)

    def reset(self):
        """
        Resets this sampler. Removes all mujoco objects from this sampler.
        """
        self.mujoco_objects = []

    def sample(self, fixtures=None, reference=None, on_top=True):
        """
        Uniformly sample on a surface (not necessarily table surface).

        Args:
            fixtures (dict): dictionary of current object placements in the scene as well as any other relevant
                obstacles that should not be in contact with newly sampled objects. Used to make sure newly
                generated placements are valid. Should be object names mapped to (pos, quat, MujocoObject)

            reference (str or 3-tuple or None): if provided, sample relative placement. Can either be a string, which
                corresponds to an existing object found in @fixtures, or a direct (x,y,z) value. If None, will sample
                relative to this sampler's `'reference_pos'` value.

            on_top (bool): if True, sample placement on top of the reference object.

        Return:
            dict: dictionary of all object placements, mapping object_names to (pos, quat, obj), including the
                placements specified in @fixtures. Note quat is in (w,x,y,z) form
        """
        raise NotImplementedError


class UniformRandomSampler(ObjectPositionSampler):
    """
    Places multiple objects within the table uniformly at random, considering their sizes (radii).
    """

    def __init__(
            self,
            name,
            objects=None,  # List of object dictionaries
            x_range=(0, 0),
            y_range=(0, 0),
            rotation=None,
            rotation_axis="z",
            ensure_object_boundary_in_range=True,
            ensure_valid_placement=True,
            reference_pos=(0, 0, 0),
            z_offset=0.0,
    ):
        self.x_range = x_range
        self.y_range = y_range
        self.rotation = rotation
        self.rotation_axis = rotation_axis
        self.objects = objects if objects is not None else [
        ]  # List of objects with their properties

        super().__init__(
            name=name,
            ensure_object_boundary_in_range=ensure_object_boundary_in_range,
            ensure_valid_placement=ensure_valid_placement,
            reference_pos=reference_pos,
            z_offset=z_offset,
        )

    def _sample_x(self, radius):
        """
        Samples the x location for a given object based on its radius.

        Args:
            radius (float): Radius of the object currently being sampled for.

        Returns:
            float: sampled x position.
        """
        minimum, maximum = self.x_range
        if self.ensure_object_boundary_in_range:
            minimum += radius
            maximum -= radius
        return np.random.uniform(high=maximum, low=minimum)

    def _sample_y(self, radius):
        """
        Samples the y location for a given object based on its radius.

        Args:
            radius (float): Radius of the object currently being sampled for.

        Returns:
            float: sampled y position.
        """
        minimum, maximum = self.y_range
        if self.ensure_object_boundary_in_range:
            minimum += radius
            maximum -= radius
        return np.random.uniform(high=maximum, low=minimum)

    def _sample_quat(self):
        """
        Samples the orientation for a given object.

        Returns:
            np.array: sampled object quaternion in (w, x, y, z) form.
        """
        if self.rotation is None:
            rot_angle = np.random.uniform(high=2 * np.pi, low=0)
        elif isinstance(self.rotation, collections.abc.Iterable):
            rot_angle = np.random.uniform(high=max(self.rotation),
                                          low=min(self.rotation))
        else:
            rot_angle = self.rotation

        if self.rotation_axis == "x":
            return np.array(
                [np.cos(rot_angle / 2),
                 np.sin(rot_angle / 2), 0, 0])
        elif self.rotation_axis == "y":
            return np.array(
                [np.cos(rot_angle / 2), 0,
                 np.sin(rot_angle / 2), 0])
        elif self.rotation_axis == "z":
            return np.array(
                [np.cos(rot_angle / 2), 0, 0,
                 np.sin(rot_angle / 2)])
        else:
            raise ValueError(
                "Invalid rotation axis specified. Must be 'x', 'y', or 'z'.")

    def sample(self, fixtures=None, reference=None, on_top=True):
        """
        Uniformly sample relative to this sampler's reference_pos or @reference (if specified), taking into account object radius.

        Args:
            fixtures (dict): Current object placements in the scene.
            reference (str or 3-tuple or None): Sample relative placement.
            on_top (bool): If True, sample placement on top of the reference object.

        Return:
            dict: Dictionary of all object placements.
        """
        placed_objects = {} if fixtures is None else copy(fixtures)
        placed_result = {}
        if reference is None:
            base_offset = self.reference_pos
        elif isinstance(reference, str):
            assert reference in placed_objects, f"Invalid reference received: {reference}"
            ref_pos, _, ref_obj = placed_objects[reference]
            base_offset = np.array(ref_pos)
            if on_top:
                base_offset += np.array((0, 0, ref_obj.top_offset[-1]))
        else:
            base_offset = np.array(reference)
            assert base_offset.shape[
                0] == 3, "Reference should be a (x,y,z) 3-tuple."

        for obj in self.objects:
            assert obj[
                "name"] not in placed_objects, f"Object '{obj['name']}' has already been sampled!"
            radius = obj["radius"]
            bottom_offset = obj["bottom_offset"]

            success = False
            for _ in range(5000):  # Retry placement up to 5000 times
                object_x = self._sample_x(radius) + base_offset[0]
                object_y = self._sample_y(radius) + base_offset[1]
                object_z = self.z_offset + base_offset[2]
                if on_top:
                    object_z -= bottom_offset

                location_valid = True
                if self.ensure_valid_placement:
                    for (x, y, z), _, other_obj in placed_objects.values():
                        if np.linalg.norm(
                            (object_x - x,
                             object_y - y)) <= other_obj["radius"] + radius:
                            location_valid = False
                            break

                if location_valid:
                    quat = self._sample_quat()
                    pos = (object_x, object_y, object_z)
                    placed_objects[obj["name"]] = (pos, quat, obj)

                    placed_result[obj["name"]] = np.concatenate(
                        [np.array(pos), np.array(quat)])
                    success = True
                    break

            if not success:
                raise ValueError("sample failed to find a valid placement")

        # return placed_objects, placed_result

        return placed_result


class SequentialCompositeSampler(ObjectPositionSampler):
    """
    Samples position for each object sequentially, allowing chaining
    multiple placement initializers together for relative object placements.
    """

    def __init__(self, name):
        self.samplers = collections.OrderedDict()  # Sequential samplers
        self.sample_args = collections.OrderedDict()  # Args for samplers

        super().__init__(name=name)

    def append_sampler(self, sampler, sample_args=None):
        """
        Adds a new placement initializer with corresponding sampler and arguments.

        Args:
            sampler (ObjectPositionSampler): The sampler to add.
            sample_args (dict or None): Additional arguments to pass to the sampler's sample() call.
        """

        self.samplers[sampler.name] = sampler
        self.sample_args[sampler.name] = sample_args

    def hide(self, objects):
        """
        Helper method to remove an object from the workspace.

        Args:
            objects (list of dicts): List of object dictionaries to hide.
        """
        sampler = UniformRandomSampler(
            name="HideSampler",
            objects=objects,
            x_range=[-10, -20],
            y_range=[-10, -20],
            rotation=[0, 0],
            rotation_axis="z",
            z_offset=10,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=False,
        )
        self.append_sampler(sampler=sampler)

    def sample(self, fixtures=None, reference=None, on_top=True):
        """
        Sample from each placement initializer sequentially in the order
        they were appended.

        Args:
            fixtures (dict): Current object placements in the scene.
            reference (str or 3-tuple or None): Sample relative placement.
            on_top (bool): If True, sample placement on top of the reference object.

        Return:
            dict: Dictionary of all object placements.
        """
        placed_objects = {} if fixtures is None else copy(fixtures)
        placed_result = {}

        for sampler, s_args in zip(self.samplers.values(),
                                   self.sample_args.values()):
            if s_args is None:
                s_args = {}

            for arg_name, arg in zip(("reference", "on_top"),
                                     (reference, on_top)):
                if arg_name not in s_args:
                    s_args[arg_name] = arg

            new_placements = sampler.sample(fixtures=placed_objects, **s_args)
            # placed_objects.update(new_placements)
            placed_result.update(new_placements)

        return placed_result


import torch


class SequentialSampleObjects:

    def __init__(self, shelf_info, object_bbox, num_deformable_objects,
                 device) -> None:
        self.shelf_info = shelf_info

        self.object_bbox = object_bbox
        self.num_objects = self.object_bbox.shape[0]  # Number of objects
        self.num_deformable_objects = num_deformable_objects
        self.device = device
        self.init_shelf_info()

    def init_shelf_info(self):

        self.shelf_size = torch.tensor(self.shelf_info["size"]).to(self.device)
        self.pos = torch.tensor(self.shelf_info["pos"]).to(self.device)

        # Define the bounds for the x and y coordinates
        self.shelf_x_min, self.shelf_x_max = self.pos[
            0] - self.shelf_size[0] / 2, self.pos[0] + self.shelf_size[0] / 2
        self.shelf_y_min, self.shelf_y_max = self.pos[
            1] - self.shelf_size[1] / 2, self.pos[1] + self.shelf_size[1] / 2

    def randomize_orientation(self):
        # Generate random orientations (angles in radians) along the z-axis
        random_angles = ((torch.rand(self.num_objects) - 0.5) * torch.pi).to(
            self.device) * 0.0

        target_orientation = torch.cat([
            torch.zeros(self.num_objects, 2).to(self.device),
            random_angles.unsqueeze(1)
        ],
                                       dim=1)
        target_quat = math_utils.quat_from_euler_xyz(target_orientation[:, 0],
                                                     target_orientation[:, 1],
                                                     target_orientation[:, 2])
        transformed_corner = math_utils.transform_points(
            self.object_bbox.view(self.num_objects, -1, 3).clone(),
            torch.zeros((self.num_objects, 3)).to(self.device),
            target_quat,
        ).reshape(self.num_objects, self.object_bbox.shape[1])

        max_bbox = transformed_corner.reshape(self.num_objects, -1,
                                              3).max(dim=1).values
        min_bbox = transformed_corner.reshape(self.num_objects, -1,
                                              3).min(dim=1).values
        transformed_size = max_bbox - min_bbox
        transformed_bbox = torch.cat([max_bbox, min_bbox], dim=1)

        return transformed_size, transformed_bbox, target_quat

    def sample(self):
        transformed_size, transformed_bbox, target_quat = self.randomize_orientation(
        )

        # Generate the shuffled indices
        shuffled_indices = torch.randperm(self.num_objects)

        # Create the inverse permutation to return to original order
        inverse_permutation = torch.argsort(shuffled_indices)
        #shuffled_indices = torch.arange(self.num_objects).to(self.device)

        y_range = transformed_size[shuffled_indices, 1]

        # Calculate the cumulative sum for y-axis positioning

        cumulative_y_range = torch.cumsum(y_range, dim=0) - y_range / 2 + 0.05

        # Create an empty tensor to store the sampled positions
        sampled_positions = torch.zeros((self.num_objects, 2)).to(self.device)
        sampled_positions[:, 1] = self.shelf_y_min + cumulative_y_range

        sampled_positions[:, 0] = (self.shelf_x_max - abs(
            torch.max(transformed_bbox[shuffled_indices][:, [0, 3]],
                      dim=1).values))
        # Reorder sampled_positions back to the original order using the inverse permutation
        original_order_positions = sampled_positions[inverse_permutation]

        postion = torch.cat(
            [original_order_positions, -transformed_bbox[:, -1].unsqueeze(1)],
            dim=1)

        result = torch.cat([postion, target_quat], dim=1)

        return result
