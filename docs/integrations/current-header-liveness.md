# Current file identity and model presentation

Passive `Models.get_models_by_path` presentation can correct a cached family when
the current native header identifies a more specific supported architecture.
Publisher title, IDs, provenance and the original metadata file stay intact.
This correction now requires the file signature after inspection to match the
signature captured before inspection. A replaced or removed source cannot receive
the just-computed family label. Exact prepared-source proofs retain their separate
current artifact/signature checks. No hash or publisher request is added.

Two existing caller fixtures also reflected older contracts. Chat's default
`selection_choices(..., preserve=True)` keeps an unavailable saved draft visible
and disabled while the loaded runtime remains unchanged. Explicit `preserve=False`
chooses an actual available quantization. Native Hunyuan image classification
corrects a stale SDXL family label as well as the older shared Hunyuan Video label;
it must still preserve file identity and presentation provenance.

Both obsolete assertions failed on the unchanged `0cb18893` CPU snapshot before
these corrections. The later mutation barrier in the header fixture exposed an
actual race and remains enforced. Source-bound comparison receipts are
`cache/plan-execution-20261111/W02-current-reader-caller-baseline-0cb18893.log`
and `W02-current-reader-caller-baseline-source.json`. These are CPU presentation
checks, independent of the separate full-app fatal-error acceptance.

```sh
python tools/check_cpu.py test_model_header_detection.py test_chat_catalogue.py test_prepared_native_source.py test_local_model_refresh.py
```
