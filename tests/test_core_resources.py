"""Core services must include the bundled data they read during construction."""

import unittest

from ruinedfooocus.core.translation_manager import TranslationManager


class CoreResourcesTests(unittest.TestCase):
    def test_translation_manager_loads_its_packaged_language_data(self):
        manager = TranslationManager()
        self.assertIn("en", manager.translations)
        manager.set_language("en")
        self.assertEqual(manager.translate("Unregistered test phrase"), "Unregistered test phrase")
