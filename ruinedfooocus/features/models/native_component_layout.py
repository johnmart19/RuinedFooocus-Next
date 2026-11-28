"""Source-bound Layered component roles shared by inventory and runtime loading."""

from dataclasses import dataclass

from ruinedfooocus.features.models.model_identity import file_signature
from ruinedfooocus.features.models.native_recipes import current_native_recipe
from ruinedfooocus.features.models.resource_defaults import RESOURCE_DEFAULTS

KEY = "_native_component_layout"


@dataclass(frozen=True)
class NativeComponentLayout:
    family: str
    clip_roles: tuple[str, ...]
    vae_role: str
    recipe_id: str
    identity_basis: str
    signature: tuple

    def verify(self, path, family=None):
        if (
            tuple(file_signature(path)) != self.signature
            or family is not None
            and family != self.family
        ):
            raise ValueError("The captured component recipe no longer matches this checkpoint.")

    def metadata(self):
        return dict(
            family=self.family,
            clip_roles=list(self.clip_roles),
            vae_role=self.vae_role,
            recipe_id=self.recipe_id,
            identity_basis=self.identity_basis,
        )


def current_layout(path, manager, *, family=None):
    recipe = current_native_recipe(path, manager)
    if recipe.get("id") != "native-qwenimage-layered":
        return None
    contract = recipe.get("applicability", {}).get("component_contract", {})
    if recipe.get("media") != "image" or not contract:
        return None
    actual_family = recipe.get("family")
    if family is not None and family != actual_family:
        return None
    if contract.get("native_family") != actual_family:
        raise ValueError("The native component contract names a different model family.")
    clips, vae = contract.get("text_encoder_roles"), contract.get("vae_role")
    if not isinstance(clips, list) or any(
        not isinstance(role, str) or not role.startswith("clip_") or role not in RESOURCE_DEFAULTS
        for role in clips
    ):
        raise ValueError("The current native recipe has no supported encoder role layout.")
    if (
        not isinstance(vae, str)
        or vae != "pixel_space"
        and (not vae.startswith("vae_") or vae not in RESOURCE_DEFAULTS)
    ):
        raise ValueError("The current native recipe has no supported VAE role layout.")
    return NativeComponentLayout(
        actual_family,
        tuple(clips),
        vae,
        recipe["id"],
        recipe.get("identity_basis", "current_native_recipe"),
        tuple(file_signature(path)),
    )
