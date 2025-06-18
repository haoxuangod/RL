import os
from tensorboard.backend.event_processing import event_accumulator
from torch.utils.tensorboard import SummaryWriter



class TBLogCollector:
    """
    Traverses a TensorBoard log root directory (including subdirectories),
    and collects all event types (scalars, histograms, images, audio, text, tensors).

    Attributes:
        log_root (str): Root directory containing TensorBoard runs.
    """
    def __init__(self, log_root: str):
        self.log_root = log_root

    def collect(self) -> dict:
        """
        Walk through log_root and its subdirectories, read all event files,
        and collect events by run_name, category, and tag.

        Returns:
            logs (Dict[str, Dict[str, Dict[str, List]]]):
                Mapping from run_name -> category -> {tag: list of event tuples}.
        """
        logs = {}
        for root, _, files in os.walk(self.log_root):
            # find event files
            event_files = [f for f in files if f.startswith("events.out.tfevents.")]
            if not event_files:
                continue

            rel = os.path.relpath(root, self.log_root)
            #run_name = rel.replace(os.sep, '.') if rel != '.' else 'root'
            run_name=rel
            logs.setdefault(run_name, {
                'scalars': {},
                'histograms': {},
                'images': {},
                'audio': {},
                'text': {},
                'tensors': {}
            })

            ea = event_accumulator.EventAccumulator(
                root,
                size_guidance={c: 0 for c in ['scalars', 'histograms', 'images', 'audio', 'text', 'tensors']}
            )
            ea.Reload()

            for category, tags in ea.Tags().items():
                if category not in logs[run_name]:
                    continue
                # tags should be iterable of tag names
                for tag in tags:
                    # ensure tag is hashable (string)
                    if not isinstance(tag, str):
                        continue
                    container = logs[run_name][category]
                    # initialize list for this tag
                    container.setdefault(tag, [])

                    # collect based on category
                    if category == 'scalars':
                        for e in ea.Scalars(tag):
                            container[tag].append((e.step, e.value))
                    elif category == 'histograms':
                        for h in ea.Histograms(tag):
                            container[tag].append((h.step, h.histogram))
                    elif category == 'images':
                        for im in ea.Images(tag):
                            container[tag].append((im.step, im.encoded_image_string))
                    elif category == 'audio':
                        for au in ea.Audios(tag):
                            container[tag].append((au.step, au.encoded_audio_string))
                    elif category == 'text':
                        for tx in ea.Text(tag):
                            # extract string from summary
                            text_val = None
                            for v in tx.summary.value:
                                if hasattr(v, 'tag') and v.tag == tag:
                                    text_val = v.simple_value if hasattr(v, 'simple_value') else v.tensor.tensor_content
                            # fallback to first
                            text_str = text_val or tx.summary.value[0].simple_value if tx.summary.value else ''
                            container[tag].append((tx.step, text_str))
                    elif category == 'tensors':
                        for ten in ea.Tensors(tag):
                            container[tag].append((ten.step, ten.tensor_proto))
        return logs


def write_logs_to_dir(logs: dict, output_root: str):
    """
    Writes the collected logs dict to a new TensorBoard directory structure,
    preserving run directories and event types.

    Args:
        logs (Dict[str, Dict[str, Dict[str, List]]]):
            Output from TBLogCollector.collect().
        output_root (str):
            Root path where new event directories will be created.
    """
    for run_name, categories in logs.items():
        run_dir = os.path.join(output_root, run_name)
        os.makedirs(run_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=run_dir)

        # Replay all categories
        for tag, events in categories.get('scalars', {}).items():
            for step, value in events:
                writer.add_scalar(tag, value, step)
        for tag, events in categories.get('histograms', {}).items():
            for step, hist in events:
                writer.add_histogram(tag, hist, step)
        for tag, events in categories.get('images', {}).items():
            for step, img in events:
                writer.add_image(tag, img, step)
        for tag, events in categories.get('audio', {}).items():
            for step, audio in events:
                writer.add_audio(tag, audio, step)
        for tag, events in categories.get('text', {}).items():
            for step, text in events:
                writer.add_text(tag, text.decode('utf-8', errors='ignore'), step)
        for tag, events in categories.get('tensors', {}).items():
            for step, tensor in events:
                writer.add_embedding(
                    tensor=tensor,
                    global_step=step,
                    tag=tag
                )

        writer.close()