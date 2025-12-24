import unillm
from ..settings import SETTINGS


def new_llm_client(model: str, is_async: bool = False, **kwargs):
    if is_async:
        return unillm.AsyncUnify(
            model,
            cache=SETTINGS.UNILLM_CACHE,
            service_tier=SETTINGS.UNILLM_SERVICE_TIER,
            **kwargs,
        )
    return unillm.Unify(
        model,
        cache=SETTINGS.UNILLM_CACHE,
        service_tier=SETTINGS.UNILLM_SERVICE_TIER,
        **kwargs,
    )
