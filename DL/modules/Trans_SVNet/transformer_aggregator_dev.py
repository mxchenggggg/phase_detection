from modules.mastoid.mastoid_module_base import MastoidModuleBase
from modules.mastoid.mastoid_predictions_callback_base import MastoidPredictionsCallbackBase
from typing import List, Dict


class TransSVNetTransAggPredClbk(
        MastoidPredictionsCallbackBase):
    def _split_predictions_outputs_by_videos(
            self, module: MastoidModuleBase, pred_outputs: List) -> Dict:
        vid_indexes = module.datamodule.vid_idxes["pred"]
        outputs_by_videos = {}
        for idx, outputs in enumerate(pred_outputs):
            outputs_by_videos[vid_indexes[idx]] = {
                "preds": outputs["preds"], "targets": outputs["targets"]}

        return outputs_by_videos
