import torch
import torch.nn as nn

d = 1e-16  # used for avoiding zero division


class BEVTransform(nn.Module):

    @staticmethod
    def get_bev_param(input_size, h_cam, p_cam, fu, fv=None, plane_h=0.,
                      w2i=True
                      ):
        """
        :param input_size: (in_height, in_width)
        :param plane_h: float, target plane height
        :param h_cam: (B, ) tensor of camera altitude
        :param p_cam: (B, ) tensor of camera pitch angle
        :param fu: (B, ) tensor of camera horizontal focal length
        :param fv: (B, ) tensor of camera vertical focal length, if None,
                        it will be set to camera_fu
        :param w2i: if true, the returned transformation is from the world frame
                    to the the image frame, otherwise, from the image frame to
                    the world frame.
        :return transformation matrices (B, 3, 3) tensor;
                BEV centers (B, 3) tensor;
                BEV scales (B, ) tensor
        """

        # unpack params
        batch_size = h_cam.size(0)
        in_height, in_width = input_size
        cu, cv = in_width / 2., in_height / 2.
        if fv is None:
            fv = fu
        h_ref, p = h_cam - plane_h, p_cam

        sin_p = torch.sin(p)
        cos_p = torch.cos(p)

        mats = h_cam.new_zeros((batch_size, 3, 3))
        if w2i:  # world to image
            # homogeneous transformation
            # /         cu cos(beta),        -fu,          cu h sin(beta)         \
            # |                                                                   |
            # | cv cos(beta) - fv sin(beta),  0,  h (fv cos(beta) + cv sin(beta)) |
            # |                                                                   |
            # \          cos(beta),           0,            h sin(beta)           /
            mats[:, 0, 0] = cu * cos_p
            mats[:, 0, 1] = -fu
            mats[:, 0, 2] = cu * h_ref * sin_p
            mats[:, 1, 0] = cv * cos_p - fv * sin_p
            mats[:, 1, 2] = h_ref * (fv * cos_p + cv * sin_p)
            mats[:, 2, 0] = cos_p
            mats[:, 2, 2] = h_ref * sin_p
        else:
            # inverse homogeneous transformation
            # /         sin(beta)   fv cos(beta) + cv sin(beta)  \
            # |   0,  - ---------,  ---------------------------  |
            # |             fv                   fv              |
            # |                                                  |
            # |    1                             cu              |
            # | - --,      0,                    --              |
            # |   fu                             fu              |
            # |                                                  |
            # |        cos(beta)     cv cos(beta) - fv sin(beta) |
            # |   0,   ---------,  - --------------------------- |
            # \           fv h                   fv h            /
            fv_tmp = fv + d
            fu_tmp = fu + d
            h_ref_tmp = h_ref + d
            fv_href_tmp = fv * h_ref + d
            mats[:, 0, 1] = -sin_p / fv_tmp
            mats[:, 0, 2] = cv * sin_p / fv_tmp + cos_p
            mats[:, 1, 0] = -1. / fu_tmp
            mats[:, 1, 2] = cu / fu_tmp
            mats[:, 2, 1] = cos_p / fv_href_tmp
            mats[:, 2, 2] = sin_p / h_ref_tmp - cv * cos_p / fv_href_tmp

        # get centers and scales
        scales, centers = BEVTransform.get_bev_scales_and_centers(
            in_height, h_cam, p_cam, fv, sin_p, cos_p
        )
        return mats, scales, centers

    @staticmethod
    def get_bev_scales_and_centers(in_height, h_cam, p_cam, fv,
                                   sin_p=None, cos_p=None
                                   ):
        batch_size = h_cam.size(0)
        cv = in_height / 2.
        # centers and scales are determined by the ground plane
        if sin_p is None:
            sin_p = torch.sin(p_cam)
        if cos_p is None:
            cos_p = torch.cos(p_cam)
        tmp = sin_p * (cv * cos_p + fv * sin_p)
        scales = h_cam / (tmp + d)
        centers = h_cam.new_zeros([batch_size, 3])
        centers[:, 0] = h_cam * cos_p / (sin_p + d)
        centers[:, 2] = 1.
        return scales, centers

    @staticmethod
    def _make_grids(bev_size, batch_size):
        """
        :param bev_size: (height, width) of the bev array
        :param batch_size:
        :return: (B, 3, N) tensor, world coordinates of the bev grid
        """
        h, w = bev_size

        # test: 2.47s
        bev_u = torch.arange(w, dtype=torch.float32)
        bev_v = torch.arange(h, dtype=torch.float32)
        bev_v, bev_u = torch.meshgrid([bev_v, bev_u])
        bev_u = torch.flatten(bev_u)
        bev_v = torch.flatten(bev_v)
        ones = torch.ones_like(bev_u)
        bev_grid = torch.stack([bev_u, bev_v, ones], 0)
        bev_grid = bev_grid.unsqueeze(0).repeat([batch_size, 1, 1])  # (B, 3, N)
        return bev_grid

    @staticmethod
    def _interpolate(input_maps, sampled_grids):
        B, C, v_max, u_max = input_maps.size()
        # sampled_grids.size() = [B, n_channels, N], N = u_max * v_max
        u = torch.flatten(sampled_grids[:, 0, :])  # (BN, )
        v = torch.flatten(sampled_grids[:, 1, :])
        u0 = torch.floor(u).long()
        u1 = u0 + 1
        v0 = torch.floor(v).long()
        v1 = v0 + 1
        # clamp
        u0 = torch.clamp(u0, 0, u_max - 1)
        u1 = torch.clamp(u1, 0, u_max - 1)
        v0 = torch.clamp(v0, 0, v_max - 1)
        v1 = torch.clamp(v1, 0, v_max - 1)

        flat_output_size = sampled_grids.size(-1)
        pixels_batch = torch.arange(0, B) * flat_output_size
        pixels_batch = pixels_batch.view(B, 1).to(input_maps.device)
        base = pixels_batch.repeat([1, flat_output_size])  # (B, N)
        base = torch.flatten(base)  # (BN, )
        base_v0 = base + v0 * u_max
        base_v1 = base + v1 * u_max
        #    u0  u1
        # v0 [a, c],
        # v1 [b, d]
        indices_a = base_v0 + u0  # (BN, )
        indices_b = base_v1 + u0
        indices_c = base_v0 + u1
        indices_d = base_v1 + u1

        flat_maps = torch.transpose(input_maps, 0, 1).reshape(C, -1)  # (C, BN)
        pixel_values_a = flat_maps[:, indices_a]  # (C, BN)
        pixel_values_b = flat_maps[:, indices_b]
        pixel_values_c = flat_maps[:, indices_c]
        pixel_values_d = flat_maps[:, indices_d]

        area_a = (v1.float() - v) * (u1.float() - u)  # (BN, )
        area_b = (v - v0.float()) * (u1.float() - u)
        area_c = (v1.float() - v) * (u - u0.float())
        area_d = (v - v0.float()) * (u - u0.float())

        values_a = area_a * pixel_values_a
        values_b = area_b * pixel_values_b
        values_c = area_c * pixel_values_c
        values_d = area_d * pixel_values_d
        values = values_a + values_b + values_c + values_d
        values = torch.transpose(values.view(C, B, flat_output_size), 0, 1)
        return values  # (B, C, N)

    def forward(self, x, plane_height, h_cam, p_cam, fu, fv=None, i2b=True):
        """ s is the ratio between the size of x and its original map size"""
        B, C, H, W = x.size()
        input_size = bev_size = (H, W)
        # generate sampled grids
        grids = self._make_grids(bev_size, B).to(x.device)  # (B, 3, N)
        if i2b:  # image view to bird's eye view
            # calculate param of output map
            homo_mats, scales, centers = self.get_bev_param(
                input_size, h_cam, p_cam, fu, fv, plane_height, w2i=True
            )  # bev to image view backward sampling
            # bev grids to world grids
            world_grids = self.bev_coord_to_world_coord(  # (B, 3, N)
                bev_size, grids, scales, centers
            )
            sample_grids = self.world_coord_to_image_coord(  # (B, 3, N)
                world_grids, homo_mats
            )
        else:  # bird's eye view to image view
            homo_mats, scales, centers = self.get_bev_param(
                input_size, h_cam, p_cam, fu, fv, plane_height, w2i=False
            )  # image view to bev backward sampling
            # image grids to world grids
            world_grids = self.image_coord_to_world_coord(grids, homo_mats)
            sample_grids = self.world_coord_to_bev_coord(
                bev_size, world_grids, scales, centers
            )
        # interpolate
        interpolated_maps = self._interpolate(x, sample_grids)
        interpolated_maps = interpolated_maps.view(B, C, *bev_size)
        return interpolated_maps, scales, centers

    def get_iv_roi(self, bev_size, h_cam, p_cam, fu, fv, filled=False):
        if filled:
            bev_roi_mask = torch.ones(bev_size)
        else:
            bev_roi_mask = torch.zeros(bev_size)
            t = 5
            bev_roi_mask[:, :, :t, :] = 1.
            bev_roi_mask[:, :, :, :t] = 1.
            bev_roi_mask[:, :, -t:, :] = 1.
            bev_roi_mask[:, :, :, -t:] = 1.
        bev_roi_mask = bev_roi_mask.to(h_cam.device)
        iv_roi_mask = self.forward(
            bev_roi_mask, 0, h_cam, p_cam, fu, fv, i2b=False
        )[0].gt(0.5)
        return iv_roi_mask

    @staticmethod
    def image_coord_to_world_coord(pixels, homo_inv_mats):
        """ pixels should be (B, 3, N) tensor """
        bs = pixels.size(0)
        xy = homo_inv_mats @ pixels
        w = torch.reciprocal(xy[:, 2, :]).view(bs, 1, -1)
        xy *= w
        return xy

    @staticmethod
    def world_coord_to_image_coord(positions, homo_mats):
        bs = positions.size(0)
        if positions.size(1) == 2:
            ones = positions.new_ones(bs, 1, positions.size(2))
            positions = torch.cat([positions, ones], dim=1)
        uv = homo_mats @ positions
        w = torch.reciprocal(uv[:, 2, :]).view(bs, 1, -1)
        uv *= w
        return uv

    @staticmethod
    def world_coord_to_bev_coord(bev_size, positions, scales, centers):
        """
        :param bev_size: (height, width) of the bev array
        :param positions: (B, 3, N) tensor, world coordinates of positions
        :param scales: (B, ) tensor, scales of one pixel in the bev
        :param centers: (B, 2) tensor, world position of the bev center
        :return: transformed positions in the bev frame
        """
        bs = positions.size(0)
        height, width = bev_size
        bev_cu, bev_cv = width / 2., height / 2.
        mat = scales.new_zeros((bs, 3, 3))
        mat[:, 0, 1] = -1. / scales
        mat[:, 0, 2] = bev_cu + centers[:, 1] / scales
        mat[:, 1, 0] = -1. / scales
        mat[:, 1, 2] = bev_cv + centers[:, 0] / scales
        mat[:, 2, 2] = 1
        return mat @ positions

    @staticmethod
    def bev_coord_to_world_coord(bev_size, bev_pixels, scales, centers):
        """
        :param bev_size: (height, width) of the bev array
        :param bev_pixels: (B, 3, N) tensor, bev coordinates of positions
        :param scales: (B, ) tensor, scales of one pixel in the bev
        :param centers: (B, 2) or (B, 3) tensor, world position of the bev center
        :return: transformed positions in the world frame, (B, 3, N) tensor
        """
        height, width = bev_size
        bev_cu, bev_cv = width / 2., height / 2.

        s = scales.unsqueeze(1)  # (B, 1)
        cx, cy = centers[:, 0].unsqueeze(1), centers[:, 1].unsqueeze(1)
        x = cx + s * (bev_cv - bev_pixels[:, 1, :])
        y = cy + s * (bev_cu - bev_pixels[:, 0, :])
        x = x.unsqueeze(1)  # (B, 1, N)
        y = y.unsqueeze(1)  # (B, 1, N)
        return torch.cat([x, y, torch.ones_like(x)], dim=1)
