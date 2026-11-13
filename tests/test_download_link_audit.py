"""Download audits must inventory honestly and never transfer artifact bodies."""

import ast
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError
from urllib.request import HTTPSHandler, build_opener

from tools.audit_download_links import (
    Probe,
    MAX_REDIRECTS,
    NoAutomaticRedirect,
    audit,
    classify,
    inventory,
    literal,
    normalize_endpoint_result,
    public_url,
    sanitize_report_value,
)


class Response:
    def __init__(self, status=200, headers=None, body=b"", url="https://example.org/model.bin"):
        self.status = status
        self.code = status
        self.msg = "test response"
        self.headers = headers or {}
        self.body = BytesIO(body)
        self.url = url
        self.read_calls = []

    def geturl(self):
        return self.url

    def info(self):
        return self.headers

    def close(self):
        self.body.close()

    @property
    def closed(self):
        return self.body.closed

    def read(self, size=-1):
        self.read_calls.append(size)
        return self.body.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class FakeHTTPS(HTTPSHandler):
    """Exercise urllib's real error/redirect pipeline without opening a socket."""

    def __init__(self, responses):
        super().__init__()
        self.responses = iter(responses)
        self.requests = []

    def https_open(self, request):
        self.requests.append(request)
        return next(self.responses)


class DownloadLinkAuditTests(unittest.TestCase):
    def test_offline_inventory_all_catalogue_categories_and_no_imports(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            catalog = root / "ruinedfooocus/resources/model_catalogue"
            catalog.mkdir(parents=True)
            for category in ("llm", "vae", "clip", "faceswap", "upscalers"):
                (catalog / f"{category}.json").write_text(
                    json.dumps({category: {"url": f"https://example.org/{category}.bin"}})
                )
            (root / "ruinedfooocus/example.py").write_text(
                'raise RuntimeError("must never execute")\nVERSION = "v2"\nURL = f"https://example.org/{VERSION}/model.bin"\nDYNAMIC = f"https://example.org/{target}/file.whl"\nPKG = "runtime==2.0"\n'
            )
            (root / "pip").mkdir()
            (root / "pip/modules.txt").write_text("# comment\nrembg==2.0.69\n")
            (root / "ruinedfooocus/vendor").mkdir()
            (root / "ruinedfooocus/vendor/bad.py").write_text("invalid Python syntax!!!")
            with patch(
                "tools.audit_download_links.build_opener", side_effect=AssertionError("offline")
            ):
                rows, sources = inventory(root)
            self.assertEqual(len([row for row in rows if "model_catalogue" in row["owner"]]), 5)
            self.assertTrue(any(row["url"] == "https://example.org/v2/model.bin" for row in rows))
            self.assertTrue(
                any(row["kind"] == "template" and row["status"] == "unresolved" for row in rows)
            )
            self.assertTrue(any(row.get("requirement") == "rembg==2.0.69" for row in rows))
            self.assertFalse(any("vendor" in path for path in sources))

    def test_head_never_reads_body(self):
        response = Response(headers={"Content-Length": "99999999999"})
        opener = Mock()
        opener.open.return_value = response
        result = Probe(interval=0, opener=opener)("https://example.org/model.bin")
        self.assertEqual(result["status"], "reachable")
        self.assertEqual(response.read_calls, [])
        self.assertEqual(opener.open.call_args.args[0].get_method(), "HEAD")

    def test_range_fallback_closes_without_read_even_if_ignored(self):
        head, fallback = Response(405), Response(200, body=b"not read")
        opener = Mock()
        opener.open.side_effect = [head, fallback]
        result = Probe(interval=0, opener=opener)("https://example.org/model.bin")
        self.assertEqual(result["status"], "reachable")
        request = opener.open.call_args.args[0]
        self.assertEqual(request.get_method(), "GET")
        self.assertEqual(request.get_header("Range"), "bytes=0-0")
        self.assertEqual(fallback.read_calls, [])

    def test_http_error_and_status_categories(self):
        expected = {
            200: "reachable",
            206: "reachable",
            401: "authentication_required",
            403: "access_denied",
            404: "broken",
            410: "broken",
            429: "rate_limited",
            503: "transient_failure",
            400: "http_failure",
        }
        for code, status in expected.items():
            self.assertEqual(classify(code), status)
        opener = Mock()
        opener.open.side_effect = HTTPError("https://example.org/a.bin", 403, "Forbidden", {}, None)
        self.assertEqual(
            Probe(interval=0, opener=opener)("https://example.org/a.bin")["status"], "access_denied"
        )

    def test_retry_is_bounded_and_network_exception_text_is_not_reported(self):
        opener = Mock()
        opener.open.side_effect = URLError("secret=private-proxy-value")
        with patch("tools.audit_download_links.time.sleep"):
            result = Probe(retries=2, interval=0, opener=opener)("https://example.org/a.bin")
        self.assertEqual(opener.open.call_count, 3)
        self.assertEqual(result["status"], "network_failure")
        self.assertNotIn("private-proxy-value", json.dumps(result))

    def test_metadata_is_size_bounded(self):
        response = Response(body=b"x" * (1024 * 1024 + 2))
        opener = Mock()
        opener.open.return_value = response
        result = Probe(interval=0, opener=opener)(
            "https://huggingface.co/api/models/org/repo", metadata=True
        )
        self.assertEqual(result["status"], "metadata_limit")
        self.assertEqual(response.read_calls, [1024 * 1024 + 1])

    def test_different_main_does_not_claim_payload_changed_and_metadata_is_cached(self):
        pin = "a" * 40
        rows = [
            {
                "owner": "catalog.json",
                "key": name,
                "kind": "artifact",
                "url": f"https://huggingface.co/org/repo/resolve/{pin}/{name}.gguf",
                "provider": "huggingface",
                "repository": "org/repo",
                "revision": pin,
                "status": "not_checked",
            }
            for name in ("a", "b")
        ]
        original = [row["url"] for row in rows]
        calls = []

        def probe(url, metadata=False):
            calls.append((url, metadata))
            return (
                {"status": "reachable", "data": {"sha": "b" * 40}}
                if metadata
                else {"status": "reachable"}
            )

        audit(rows, probe, workers=2, compare_metadata=True)
        self.assertEqual(sum(metadata for _, metadata in calls), 1)
        self.assertEqual([row["url"] for row in rows], original)
        self.assertTrue(
            all(
                row["status"] == "reachable"
                and row["freshness"] == "upstream_revision_differs"
                and row["artifact_change"] == "unknown"
                and not row["update_candidate"]["validated"]
                for row in rows
            )
        )

    def test_unique_url_limit_preserves_unaudited_rows_and_resolvers(self):
        rows = [
            {"url": url, "status": "not_checked", "kind": "artifact"}
            for url in (
                "https://example.org/a.bin",
                "https://example.org/a.bin",
                "https://example.org/b.bin",
            )
        ]
        rows.append({"url": None, "status": "unresolved", "kind": "resolver"})
        probe = Mock(return_value={"status": "reachable"})
        audit(rows, probe, limit=1)
        probe.assert_called_once_with("https://example.org/a.bin")
        self.assertEqual(
            [row["status"] for row in rows], ["reachable", "reachable", "not_checked", "unresolved"]
        )

    def test_signed_redirect_evidence_is_redacted(self):
        value = public_url(
            "https://username:password@example.org/file.bin?token=secret&Expires=123#private"
        )
        for secret in ("username", "password", "secret", "123", "private"):
            self.assertNotIn(secret, value)

    def test_literal_evaluation_does_not_execute_calls(self):
        expression = ast.parse('f"https://example.org/{danger()}/x.whl"', mode="eval").body
        self.assertEqual(literal(expression, {}), "https://example.org/{danger()}/x.whl")

    def test_symbolic_port_is_preserved_in_unresolved_template(self):
        self.assertEqual(public_url("http://127.0.0.1:{port}/"), "http://127.0.0.1:{port}/")

    def test_concrete_filename_path_is_encoded_without_double_encoding(self):
        self.assertEqual(
            public_url("https://example.org/模型/a b%20c.bin"),
            "https://example.org/%E6%A8%A1%E5%9E%8B/a%20b%20c.bin",
        )

    def test_worker_failure_is_saved_and_progress_continues(self):
        rows = [
            {
                "url": f"https://example.org/{number}.bin",
                "kind": "artifact",
                "status": "not_checked",
            }
            for number in range(27)
        ]
        progress = []

        def probe(url):
            if url.endswith("/3.bin"):
                raise RuntimeError("credential=must-not-leak")
            return {"status": "reachable"}

        audit(rows, probe, progress=progress.append)
        self.assertEqual(rows[3]["status"], "probe_failure")
        self.assertEqual(sum(row["status"] == "reachable" for row in rows), 26)
        self.assertNotIn("must-not-leak", json.dumps(rows))
        self.assertEqual([event["completed"] for event in progress], [25, 27])

    def test_metadata_worker_failure_is_saved_without_losing_availability(self):
        rows = [
            {
                "url": "https://huggingface.co/org/repo/resolve/main/a.bin",
                "kind": "artifact",
                "status": "not_checked",
                "provider": "huggingface",
                "repository": "org/repo",
                "revision": "main",
            }
        ]

        def probe(url, metadata=False):
            if metadata:
                raise RuntimeError("secret-metadata-url")
            return {"status": "reachable"}

        audit(rows, probe, compare_metadata=True)
        self.assertEqual(rows[0]["status"], "reachable")
        self.assertEqual(rows[0]["upstream_metadata"]["status"], "probe_failure")
        self.assertNotIn("secret-metadata-url", json.dumps(rows))

    def test_composed_api_base_failure_is_not_a_broken_artifact(self):
        rows = [
            {"url": "https://civitai.com/api/v1/", "kind": "endpoint", "status": "not_checked"},
            {"url": "https://example.org/model.bin", "kind": "artifact", "status": "not_checked"},
        ]

        def probe(url):
            return {"status": "broken", "evidence": {"http_status": 404}}

        audit(rows, probe)
        self.assertEqual(rows[0]["status"], "endpoint_not_found")
        self.assertEqual(rows[0]["probe_status"], "broken")
        self.assertEqual(rows[0]["evidence"]["http_status"], 404)
        self.assertIn("actual API endpoint or package artifact", rows[0]["action"])
        self.assertEqual(rows[1]["status"], "broken")

    def test_saved_index_denials_can_be_normalized_without_network(self):
        for status, code in (("authentication_required", 401), ("access_denied", 403)):
            row = {"kind": "endpoint", "status": status, "evidence": {"http_status": code}}
            normalize_endpoint_result(row)
            self.assertEqual(row["status"], "endpoint_" + status)
            self.assertEqual(row["probe_status"], status)
            self.assertEqual(row["evidence"]["http_status"], code)
            normalize_endpoint_result(row)
            self.assertEqual(row["status"], "endpoint_" + status)

    def test_real_urllib_redirect_pipeline_never_drains_get_fallback(self):
        redirect = Response(
            302,
            {"Location": "https://cdn.example.org/file.bin?token=private-signature"},
            body=b"redirect body must not be read",
        )
        final = Response(
            200,
            body=b"artifact body must not be read",
            url="https://cdn.example.org/file.bin?token=private-signature",
        )
        transport = FakeHTTPS([Response(405), redirect, final])
        probe = Probe(interval=0, opener=build_opener(NoAutomaticRedirect(), transport))
        probe.rate_limit = Mock(wraps=probe.rate_limit)
        result = probe("https://example.org/file.bin")
        self.assertEqual(result["status"], "reachable")
        self.assertEqual(
            [request.get_method() for request in transport.requests], ["HEAD", "GET", "GET"]
        )
        self.assertEqual(transport.requests[-1].get_header("Range"), "bytes=0-0")
        self.assertIn("private-signature", transport.requests[-1].full_url)
        self.assertNotIn("private-signature", json.dumps(result))
        self.assertEqual(probe.rate_limit.call_count, 3)
        for response in (redirect, final):
            self.assertEqual(response.read_calls, [])
            self.assertTrue(response.body.closed)

    def test_metadata_redirect_body_is_unread_and_final_read_is_capped(self):
        redirect = Response(307, {"Location": "/final"}, body=b"large redirect body")
        final = Response(body=b'{"sha":"abc"}')
        transport = FakeHTTPS([redirect, final])
        probe = Probe(interval=0, opener=build_opener(NoAutomaticRedirect(), transport))
        result = probe("https://example.org/api/model", metadata=True)
        self.assertEqual(result["data"], {"sha": "abc"})
        self.assertEqual(redirect.read_calls, [])
        self.assertTrue(redirect.body.closed)
        self.assertEqual(final.read_calls, [1024 * 1024 + 1])

    def test_redirect_chain_and_cycle_are_bounded_and_streams_closed(self):
        responses = [
            Response(302, {"Location": f"https://example.org/{number}"})
            for number in range(MAX_REDIRECTS + 1)
        ]
        transport = FakeHTTPS(responses)
        probe = Probe(interval=0, opener=build_opener(NoAutomaticRedirect(), transport))
        result = probe("https://example.org/start")
        self.assertEqual(result["status"], "redirect_limit")
        self.assertEqual(len(transport.requests), MAX_REDIRECTS + 1)
        self.assertTrue(
            all(not response.read_calls and response.body.closed for response in responses)
        )
        cycle = Response(302, {"Location": "https://example.org/start"})
        probe = Probe(interval=0, opener=build_opener(NoAutomaticRedirect(), FakeHTTPS([cycle])))
        self.assertEqual(probe("https://example.org/start")["status"], "redirect_loop")
        self.assertEqual(cycle.read_calls, [])
        self.assertTrue(cycle.body.closed)

    def test_redirect_rejects_tls_downgrade_and_embedded_credentials(self):
        for target in (
            "http://example.org/file.bin",
            "https://user:private-password@example.org/file.bin",
        ):
            redirect = Response(302, {"Location": target})
            probe = Probe(
                interval=0, opener=build_opener(NoAutomaticRedirect(), FakeHTTPS([redirect]))
            )
            result = probe("https://example.org/start")
            self.assertEqual(result["status"], "redirect_rejected")
            self.assertNotIn("private-password", json.dumps(result))
            self.assertEqual(redirect.read_calls, [])
            self.assertTrue(redirect.body.closed)

    def test_report_extras_redact_nested_credential_and_query_urls(self):
        private = "https://private-user:private-password@example.org/whl?token=private-token"
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "ruinedfooocus").mkdir()
            (root / "requirements_versions.txt").write_text("package @ " + private)
            (root / "ruinedfooocus/example.py").write_text(
                f'plan = RuntimeBundle("1", "2", "{private}")\n'
            )
            rows, _ = inventory(root)
        serialized = json.dumps(rows)
        for secret in ("private-user", "private-password", "private-token"):
            self.assertNotIn(secret, serialized)
        self.assertIn("<redacted>", serialized)
        value = {"requirements": ["package @ " + private], "resolver": {"index": private}}
        sanitized = sanitize_report_value(value)
        self.assertEqual(sanitize_report_value(sanitized), sanitized)


if __name__ == "__main__":
    unittest.main()
