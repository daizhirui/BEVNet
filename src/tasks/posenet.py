from pytorch_helper.task import Task
from pytorch_helper.settings.spaces import Spaces


@Spaces.register(Spaces.NAME.TASK, 'PoseNetTask')
class PoseNetTask(Task):
    def model_forward_backward(
        self, batch, backward=False
    ):
        image = batch.gt['image']
        camera_height = batch.gt['camera_height']
        camera_angle = batch.gt['camera_angle']

        pred = self.model(image)

        loss = self.loss_fn(
            pred['camera_height'], pred['camera_angle'],
            camera_height, camera_angle
        )
        if backward:
            loss['pose'].backward()

        batch.pred = pred
        batch.loss = loss
        batch.size = image.size(0)

        return batch
