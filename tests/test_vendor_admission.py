"""Driver metadata admission never imports a native runtime or runs compute."""

import unittest
from unittest.mock import Mock, patch
from ruinedfooocus.packaging import vendor_admission as admission
from ruinedfooocus.packaging.vendor_admission import combined_evidence, windows_recipe_evidence


class VendorAdmissionTests(unittest.TestCase):
    def test_bootstrap_adapter_selection_uses_one_qualified_physical_pair(self):
        old = dict(
            adapter_id="old-igpu", vendor="amd", name="AMD Radeon RX 580", pci_id="1002:67df"
        )
        good = dict(
            adapter_id="new-dgpu", vendor="amd", name="AMD Radeon RX 7900 XTX", pci_id="1002:744c"
        )
        inventory = dict(
            build="26200",
            drivers=[
                dict(DeviceID="PCI\\VEN_1002&DEV_67DF\\OLD", DriverVersion="31.0.0.0"),
                dict(DeviceID="PCI\\VEN_1002&DEV_744C\\NEW", DriverVersion="32.0.31041.5005"),
            ],
        )
        evidence = admission.windows_recipe_evidence("rocm10.1", [old, good], inventory)
        selected, hardware, driver = admission.select_installer_adapter([old, good], evidence)
        self.assertEqual(selected["inventory_adapter_id"], "new-dgpu")
        self.assertEqual(selected["hardware_id"], inventory["drivers"][1]["DeviceID"])
        self.assertTrue(hardware)
        self.assertTrue(driver)
        self.assertEqual(len(evidence), 2)
        self.assertFalse(evidence[0]["hardware_supported"])

    def test_admission_never_combines_hardware_and_driver_from_different_cards(self):
        adapters = [{"adapter_id": "a"}, {"adapter_id": "b"}]
        evidence = [
            dict(adapter_id="a", hardware_supported=True, driver_supported=False),
            dict(adapter_id="b", hardware_supported=False, driver_supported=True),
        ]
        _, hardware, driver = admission.select_installer_adapter(adapters, evidence)
        self.assertFalse(hardware is True and driver is True)

    def linux(self, recipe="rocm10.1", *, wsl=False, **changes):
        adapter = dict(
            adapter_id="native:pci:0000:03:00.0",
            name="AMD Radeon RX 7900 XTX",
            vendor="amd",
            pci_id="1002:744c",
            gfx="gfx1100",
            kernel_driver="amdgpu",
        )
        inventory = dict(
            os_release={"ID": "ubuntu", "VERSION_ID": "24.04", "PRETTY_NAME": "Ubuntu 24.04.5 LTS"},
            kernel="7.0.0-15-generic",
            packages={"amdgpu-core": "31.60.0.1234-1", "amdgpu-dkms": "1:6.16.1"},
            modules={"amdgpu": {"loaded": "same", "installed": "same"}},
            kfd=True,
            dxg=True,
            render_nodes=("/dev/dri/renderD128",),
        )
        inventory.update(changes)
        if recipe == "xpu":
            adapter.update(
                vendor="intel", name="Intel Arc B580", pci_id="8086:e20b", kernel_driver="xe"
            )
        host = dict(
            build="26200",
            drivers=[dict(DeviceID="PCI\\VEN_1002&DEV_744C\\A", DriverVersion="32.0.31041.5005")],
        )
        return admission.linux_recipe_evidence(
            recipe,
            [adapter],
            inventory,
            wsl=wsl,
            host_inventory=host,
            guest_visible={adapter["adapter_id"]: True},
        )[0]

    def test_native_linux_requires_exact_released_os_and_loaded_driver(self):
        result = self.linux()
        self.assertTrue(result["hardware_supported"])
        self.assertTrue(result["driver_supported"])
        wrong_kernel = self.linux(kernel="6.14.0-generic")
        self.assertFalse(wrong_kernel["hardware_supported"])
        wrong_module = self.linux(modules={"amdgpu": {"loaded": "old", "installed": "new"}})
        self.assertFalse(wrong_module["driver_supported"])
        unknown_branch = self.linux(packages={"amdgpu-core": "31.70.0.1", "amdgpu-dkms": "known"})
        self.assertIsNone(unknown_branch["driver_supported"])

    def test_wsl_requires_host_driver_guest_visibility_and_preview_marker(self):
        result = self.linux(wsl=True)
        self.assertTrue(result["hardware_supported"])
        self.assertTrue(result["driver_supported"])
        self.assertTrue(result["opt_in"])
        self.assertFalse(self.linux(wsl=True, dxg=False)["driver_supported"])
        adapter = dict(
            adapter_id="host:one", name="AMD Radeon RX 7900 XTX", pci_id="1002:744c", gfx="gfx1100"
        )
        result = admission.linux_recipe_evidence(
            "rocm10.1",
            [adapter],
            {"os_release": {"ID": "ubuntu", "VERSION_ID": "24.04"}, "dxg": True},
            wsl=True,
        )[0]
        self.assertIsNone(result["driver_supported"])

    def test_clean_debian_wsl_can_acquire_wheels_before_guest_runtime_exists(self):
        adapter = dict(
            adapter_id="host:one", name="AMD Radeon RX 7900 XTX", pci_id="1002:744c", gfx="gfx1100"
        )
        host = dict(
            build="26300",
            drivers=[dict(DeviceID="PCI\\VEN_1002&DEV_744C\\A", DriverVersion="32.0.32015.12012")],
        )
        result = admission.linux_recipe_evidence(
            "rocm10.1",
            [adapter],
            {"os_release": {"ID": "debian", "VERSION_ID": "13"}, "dxg": True},
            wsl=True,
            host_inventory=host,
        )[0]
        self.assertTrue(result["hardware_supported"])
        self.assertTrue(result["driver_supported"])
        self.assertTrue(result["opt_in"])
        self.assertIn("Guest execution must be verified", " ".join(result["reasons"]))

    def test_native_xpu_requires_own_loaded_module_and_userspace_runtime(self):
        packages = {"libze1": "1", "intel-level-zero-gpu": "1", "intel-opencl-icd": "1"}
        result = self.linux(
            "xpu", packages=packages, modules={"xe": {"loaded": "same", "installed": "same"}}
        )
        self.assertTrue(result["hardware_supported"])
        self.assertTrue(result["driver_supported"])
        self.assertIsNone(
            self.linux("xpu", packages={}, modules={"xe": {"loaded": "same", "installed": "same"}})[
                "driver_supported"
            ]
        )

    def test_guest_query_binds_unique_device_without_installing_helpers(self):
        adapter = dict(adapter_id="host:a", name="AMD Radeon RX 7900 XTX")
        runner = Mock(
            return_value=Mock(
                returncode=0, stdout="Platform AMD\n  Device #0: AMD Radeon RX 7900 XTX"
            )
        )
        with patch.object(admission.shutil, "which", return_value="/usr/bin/clinfo"):
            self.assertEqual(
                admission.guest_device_inventory([adapter], runner=runner), {"host:a": True}
            )
            duplicate = dict(adapter, adapter_id="host:b")
            self.assertEqual(
                admission.guest_device_inventory([adapter, duplicate], runner=runner),
                {"host:a": None, "host:b": None},
            )
        with patch.object(admission.shutil, "which", return_value=None):
            self.assertEqual(
                admission.guest_device_inventory([adapter], runner=runner), {"host:a": None}
            )
        self.assertEqual(runner.call_count, 2)

    def test_rocm_guest_uses_existing_rocminfo_without_opencl(self):
        adapter = dict(adapter_id="host:a", name="AMD Radeon RX 7900 XTX")
        runner = Mock(
            return_value=Mock(
                returncode=0, stdout="Name: gfx1100\nMarketing Name: AMD Radeon RX 7900 XTX\n"
            )
        )
        with patch.object(admission.shutil, "which", side_effect=[None, "/usr/bin/rocminfo"]):
            self.assertEqual(
                admission.guest_device_inventory([adapter], runner=runner), {"host:a": True}
            )
        self.assertEqual(runner.call_args.args[0], ["/usr/bin/rocminfo"])

    def row(
        self,
        name="AMD Radeon RX 7900 XTX",
        pci_id="1002:744c",
        driver="32.0.31041.5005",
        build="26200",
        recipe="rocm10.1",
    ):
        adapter = dict(adapter_id="physical-a", name=name, pci_id=pci_id)
        vendor, device = pci_id.split(":")
        metadata = dict(
            build=build,
            drivers=[dict(DeviceID=f"PCI\\VEN_{vendor}&DEV_{device}\\A", DriverVersion=driver)],
        )
        return windows_recipe_evidence(recipe, [adapter], metadata)[0]

    def test_released_amd_and_intel_device_driver_combinations(self):
        amd = self.row()
        self.assertTrue(amd["hardware_supported"])
        self.assertTrue(amd["driver_supported"])
        intel = self.row("Intel Arc B580", "8086:e20b", "32.0.101.8801", recipe="xpu")
        self.assertTrue(intel["hardware_supported"])
        self.assertTrue(intel["driver_supported"])

    def test_old_driver_wrong_device_and_os_are_rejected(self):
        self.assertFalse(self.row(driver="32.0.21041.5000")["driver_supported"])
        self.assertFalse(self.row(name="AMD Radeon RX 580")["hardware_supported"])
        self.assertFalse(self.row(build="22000")["hardware_supported"])
        self.assertFalse(
            self.row("Intel HD Graphics 630", "8086:5912", "32.0.101.8801", recipe="xpu")[
                "hardware_supported"
            ]
        )

    def test_new_unlisted_amd_driver_and_missing_metadata_stay_unknown(self):
        self.assertTrue(self.row(driver="32.0.32015.12012")["driver_supported"])
        self.assertIsNone(self.row(driver="33.0.1.2")["driver_supported"])
        self.assertIsNone(self.row(driver="")["driver_supported"])
        self.assertIsNone(self.row(build="")["hardware_supported"])
        self.assertIsNone(combined_evidence([True, None]))
        self.assertFalse(combined_evidence([True, False, None]))

    def test_duplicate_cards_cannot_share_one_unbound_driver_identity(self):
        adapter = dict(adapter_id="a", name="AMD Radeon RX 7900 XTX", pci_id="1002:744c")
        rows = [
            dict(DeviceID=f"PCI\\VEN_1002&DEV_744C\\{instance}", DriverVersion="32.0.31041.5005")
            for instance in ("A", "B")
        ]
        inventory = dict(build="26200", drivers=rows)
        self.assertIsNone(
            windows_recipe_evidence("rocm10.1", [adapter], inventory)[0]["driver_supported"]
        )
        adapter["hardware_id"] = rows[1]["DeviceID"]
        self.assertTrue(
            windows_recipe_evidence("rocm10.1", [adapter], inventory)[0]["driver_supported"]
        )
