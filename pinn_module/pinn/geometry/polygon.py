# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""2d geometry"""

from __future__ import absolute_import
import mindspore
from mindspore import log as logger
from mindelec.geometry.geometry_base import Geometry, GEOM_TYPES
from mindelec.geometry.utils import sample, generate_mesh
from scipy import spatial
import numpy as np


class Polygon(Geometry):
    r"""
    Definition of Polygon object.

    Args:
        name (str): name of the hyper cube.
        vertices (list): coordinates of the vertices
        dtype (numpy.dtype): Data type of sampled point data type. Default: numpy.float32.
        sampling_config (SamplingConfig): sampling configuration. Default: None

    Raises:
        TypeError: sampling_config is not instance of class SamplingConfig.
    Supported Platforms:
        ``Ascend``
    """

    def __init__(self, name, vertices):
        self.nvertices = len(vertices)
        self.vertices, self.area = polygon_area(vertices)
        self.name = name
        # to calculate the pairwise distances between any two vertex
        self.diagonals = spatial.distance.squareform(spatial.distance.pdist(self.vertices))
        super().__init__(
            name,
            2,
            [int(np.min(self.vertices[:, 0])), int(np.min(self.vertices[:, 1]))],
            [int(np.max(self.vertices[:, 0])), int(np.max(self.vertices[:, 1]))])

        self.perimeter = np.sum([self.diagonals[i, i + 1]
                                 for i in range(-1, self.nvertices - 1)])  # sum of all lengths of edges
        # calculate the normal vector of the edges
        self.normal = normal_vector(self.vertices)
        self.columns_dict = {}

    def sampling(self, geom_type="domain"):
        """
        sampling points

        Args:
            geom_type (str): geometry type

        Returns:
            Numpy.array, 2D numpy array with or without boundary normal vectors

        Raises:
            ValueError: If `config` is None.
            KeyError: If `geom_type` is `domain` but `config.domain` is None.
            KeyError: If `geom_type` is `BC` but `config.bc` is None.
            ValueError: If `geom_type` is neither `BC` nor `domain`.
        """
        config = self.sampling_config
        if config is None:
            raise ValueError("Sampling config for {}:{} is None, please call set_sampling_config method to set".format(
                self.geom_type, self.name))
        if not isinstance(geom_type, str):
            raise TypeError("geom_type shouild be string, but got {} with type {}".format(geom_type, type(geom_type)))
        if geom_type not in GEOM_TYPES:
            raise ValueError("Unsupported geom_type: {}, only {} are supported now".format(geom_type, GEOM_TYPES))

        # 采样主体
        if geom_type.lower() == "domain":
            if config.domain is None:
                raise KeyError("Sampling config for domain of {}:{} should not be none"
                               .format(self.geom_type, self.name))
            logger.info("Sampling domain points for {}:{}, config info: {}"
                        .format(self.geom_type, self.name, config.domain))
            column_name = self.name + "_domain_points"
            if config.domain.random_sampling:
                data = self._random_domain_points(config.domain.size)
            else:
                data = self._grid_domain_points(config.domain.size)
            self.columns_dict["domain"] = [column_name]
            data = data.astype(self.dtype)
            return data

        # 采样边界点
        if geom_type.lower() == "bc":
            if config.bc is None:
                raise KeyError("Sampling config for BC of {}:{} should not be none".format(self.geom_type, self.name))
            logger.info("Sampling BC points for {}:{}, config info: {}"
                        .format(self.geom_type, self.name, config.bc))
            if config.bc.with_normal:
                if config.bc.random_sampling:
                    data, data_normal = self._random_boundary_points(config.bc.size)
                else:
                    data, data_normal = self._grid_boundary_points(config.bc.size)
                column_data = self.name + "_BC_points"
                column_normal = self.name + "_BC_normal"
                self.columns_dict["BC"] = [column_data, column_normal]
                data = data.astype(self.dtype)
                data_normal = data_normal.astype(self.dtype)
                return data, data_normal

            if config.bc.random_sampling:
                data = self._random_boundary_points(config.bc.size)
            else:
                data = self._grid_boundary_points(config.bc.size)
            column_data = self.name + "_BC_points"
            self.columns_dict["BC"] = [column_data]
            data = data.astype(self.dtype)
            return data
        raise ValueError("Unknown geom_type: {}, only \"domain/BC\" are supported for {}:{}".format(
            geom_type, self.geom_type, self.name))

    def _inside(self, points):
        """whether inside domain
        winding number method.
        Args:
            P: A point.
            V: Vertex points of a polygon.
        Returns:
            judge: true means inside and false means outside.
        """

        def isleft(p0, p1, p2):
            return np.cross(p1 - p0, p2 - p0, axis=-1).reshape((1, -1))

        py = points[:, 1]
        is_in = np.zeros(len(points))

        for i in range(self.nvertices):
            condition1 = np.logical_and(np.logical_and(self.vertices[i, 1] <= py,
                                                       self.vertices[(i + 1) % self.nvertices, 1] > py),
                                        isleft(self.vertices[i, :],
                                               self.vertices[(i + 1) % self.nvertices, :], points) > 0)

            condition2 = np.logical_and(np.logical_and(self.vertices[i, 1] > py,
                                                       self.vertices[(i + 1) % self.nvertices, 1] <= py),
                                        isleft(self.vertices[i, :],
                                               self.vertices[(i + 1) % self.nvertices, :], points) < 0)
            condition1 = condition1.astype(int).reshape(-1)
            condition2 = condition2.astype(int).reshape(-1)
            is_in += condition1
            is_in -= condition2
        return is_in != 0

    def _on_boundary(self, points):
        """whether on geometry's boundary
        suppose vector a = A to P, vector b = B to P
        point P is on segment AB == a * b + |a| * |b| = 0 """
        _on = np.ones(shape=len(points), dtype=np.int) * -1
        for i in range(len(points)):
            vertices_i = self.vertices
            vertices_i[:, 0] = points[i, 0] - vertices_i[:, 0]
            vertices_i[:, 1] = points[i, 1] - vertices_i[:, 1]
            for j in range(-1, self.nvertices - 1):
                vector_a = vertices_i[j, :]
                vector_b = vertices_i[j + 1, :]
                if (np.dot(vector_a, vector_b) + np.linalg.norm(vector_a)
                        + np.linalg.norm(vector_b) == 0):
                    _on[i] = 0
                    break
        return _on == 0

    def _boundary_normal(self, points):
        """get the normal vector of boundary points"""
        for i in range(self.nvertices):
            if is_on_line_segment(self.vertices[i - 1], self.vertices[i], points):
                return self.normal[i]
        return np.array([0, 0])

    def _random_domain_points(self, n, random="uniform"):
        """randomly generate domain points"""
        x = np.empty((0, 2))
        two_d_range = np.array(
            [np.min(self.vertices, axis=0), np.max(self.vertices, axis=0)])  # 2-D range of the polygon
        range_difference = two_d_range[1, :] - two_d_range[0, :]
        # sample new boundary points repeatly until sample enough points
        while len(x) < n:
            # interval linear projection: [0, 1] to [left, right]
            x_new = sample(n, 2, sampler="uniform") * range_difference + two_d_range[0]
            new_points = x_new[self._inside(x_new)].reshape(-1, 2)
            x = np.concatenate([x, new_points], axis=0)
        # in case in last iteration, the number of new-generated points is so large that total nums is bigger than n
        return x[:n]

    def _random_boundary_points(self, n, random="uniform"):
        """get boundary points randomly"""
        # each number represents the disance between the starting vertex and the ending vertex along the edges
        # so the generated new data, multiplied by perimeter, can be projected to the required interval
        u = np.ravel(sample(n, 1, random)) * self.perimeter
        # to confirm the corresponding coordinates of the sampled point according to the random distance
        distance = np.empty((self.nvertices + 1, 1))
        distance[0] = 0
        for i in range(1, self.nvertices):
            distance[i] = self.diagonals[i - 1, i]  # the length of each edge
        distance[-1] = self.diagonals[-1, 0]
        distance = distance.cumsum(axis=0)
        x = np.empty((n, 2))
        for i in range(n):
            index = 0
            while (distance[index] < u[i]):
                index += 1
            l2 = u[i] - distance[index - 1]
            x[i, :] = self.vertices[index - 1, :] + l2 / self.diagonals[index - 1, index % self.nvertices] \
                      * (self.vertices[index % self.nvertices, :] - self.vertices[index - 1, :])
        return x


def is_on_line_segment(p0, p1, p2):
    """ Whether a point is between two other points on a line segment.
    Args:
        P0: One point in the line.
        P1: One point in the line.
        P2: The point to be tested.
    """
    v02 = p2 - p0
    v12 = p2 - p1
    return np.dot(v02, v12) + np.linalg.norm(v02) + np.linalg.norm(v12) == 0


def polygon_area(vertices):
    """The area of a polygon.
    Notice that whether the vertices are counterclockwise or clockwise
    may result in different results of Positivity and negativity.

    S1 = (x1y2+x2y3+...+xn·y1)
    S2 = (x1yn+x2y1+x3y2+...xny_{n-1})
    the final result S = (S1 - S2) / 2
    """
    vertices = np.array(vertices)
    x1 = np.array([vertices[-1, 1], vertices[-1, 1]]).reshape(1, -1)
    x2 = np.array([vertices[0, 1], vertices[0, 1]]).reshape(1, -1)
    verties = np.concatenate([x1, vertices, x2], axis=0)
    area = 0.5 * (np.dot(verties[1:-1, 0], verties[2:, 1]) - np.dot(verties[1:-1, 0], verties[:-2, 1]))
    if area < 0:
        return np.flipud(vertices), -area
    else:
        return vertices, area


def normal_vector(vertices):
    segments = vertices[1:] - vertices[:-1]
    first_edge = vertices[0] - vertices[-1]
    first_edge = first_edge.reshape(1, -1)
    # calculate the edges(vector) of the polygon
    segments = np.concatenate([first_edge, segments], axis=0)
    # suppose the normal vector n = (a,b) is the normal vector of the edge(c, d),
    # they satisfy the relationship ac + bd = 0(1)
    # we can suppose any real number of a(such as -d) and calculate b for equation(1) has infinite solutions
    # then we can just normalize the vector n to calculate the normal vector
    normal = np.empty(segments.shape)
    normal[:, 0] = segments[:, 1] * -1
    normal[:, 1] = segments[:, 0]
    return normal / np.linalg.norm(normal, axis=1).reshape(-1, 1)
