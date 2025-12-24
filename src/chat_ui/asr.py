from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess


class Fun_ASR:
    def __init__(self, model = "iic/SenseVoiceSmall", vad_model = "fsmn-vad", vad_kwargs = {"max_single_segment_time": 30000}, device = "cuda", disable_update = True):
        self.model = AutoModel(
            model = model,
            # vad_model = vad_model,
            # vad_kwargs=vad_kwargs,
            device = device,
            disable_update = disable_update,
        )

    def infer(self, audio_file):
        res = self.model.generate(
            input = audio_file,
            cache = {},
            language = "auto",
            use_itn = True,
            batch_size_s = 60,
            merge_vad = True,
            merge_length_s = 15,
        )
        text = rich_transcription_postprocess(res[0]["text"])

        return text
