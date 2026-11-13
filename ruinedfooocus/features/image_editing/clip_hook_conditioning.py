"""Use native hook intervals for conditioning nodes with a direct CLIP encode."""


def _interval(conditioning, bounds):
    result = []
    for tensor, metadata in conditioning:
        start = max(bounds[0], metadata.get("clip_start_percent", 0.0))
        end = min(bounds[1], metadata.get("clip_end_percent", 1.0))
        if max(start, metadata.get("start_percent", 0.0)) >= min(
            end, metadata.get("end_percent", 1.0)
        ):
            continue
        result.append([tensor, dict(metadata, clip_start_percent=start, clip_end_percent=end)])
    return result


def direct_conditioning(clip, native_call, conditioning=None):
    """Run the native node once per native CLIP interval, retaining its output.

    ``native_call`` accepts the interval's CLIP, and, for a conditioning transform,
    its intersected input list. Original tensors, hooks and keyframes stay owned
    by the request. Constant CLIP and Off take the original single-call path.
    """
    hooks = getattr(getattr(clip, "patcher", None), "forced_hooks", None)
    if hooks is None or not getattr(clip, "use_clip_schedule", False):
        return native_call(clip) if conditioning is None else native_call(clip, conditioning)

    from comfy.hooks import HookGroup, HookKeyframe, HookKeyframeGroup
    from comfy.model_management import throw_exception_if_processing_interrupted

    output = []
    for bounds, selected_frames in hooks.get_hooks_for_clip_schedule():
        selected = None if conditioning is None else _interval(conditioning, bounds)
        if selected == []:
            continue
        throw_exception_if_processing_interrupted()
        local = clip.clone(disable_dynamic=True)
        try:
            constant = HookGroup()
            for hook, frame in selected_frames:
                current = hook.clone()
                # Native Hook.clone shares its keyframe group. Replace that group
                # before setting this interval; model hooks may share the original.
                current.hook_keyframe = HookKeyframeGroup()
                current.hook_keyframe.add(HookKeyframe(1.0 if frame is None else frame.strength))
                constant.add(current)
            local.patcher.forced_hooks = constant
            local.use_clip_schedule = False
            result = native_call(local) if conditioning is None else native_call(local, selected)
            output.extend(_interval(result, bounds))
            throw_exception_if_processing_interrupted()
        finally:
            local.patcher.clean_hooks()
    return output
