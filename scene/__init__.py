import os

from arguments import ModelParams
from scene.dataset_readers import readData, readDatau
from scene.gaussian_model import GaussianModel
from utils.system_utils import searchForMaxIteration


class Scene:

    gaussians: GaussianModel

    def __init__(
        self,
        args: ModelParams,
        gaussians: GaussianModel,
        load_iteration=None,
        normalized=False,
        fraction=0.01
    ):
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(
                    os.path.join(self.model_path, "point_cloud")
                )
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        if os.path.exists(args.source_path) and args.source_path.lower().endswith(('.vtk', '.vtu')):
            if args.source_path.lower().endswith('.vtk'):
                mesh, pcd = readData(args.source_path, fraction, normalized=normalized)
            else:
                mesh, pcd = readDatau(args.source_path, fraction, normalized=normalized)
        else:
            assert False, "Could not recognize scene type!"

        if self.loaded_iter:
            self.gaussians.load_ply(
                os.path.join(
                    self.model_path,
                    "point_cloud",
                    "iteration_" + str(self.loaded_iter),
                    "point_cloud.ply",
                ),
                mesh
            )
        else:
            self.gaussians.create_from_pcd(
                pcd,
                mesh
            )

    def save(self, iteration):
        point_cloud_path = os.path.join(
            self.model_path, f"point_cloud/iteration_{iteration}"
        )
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))