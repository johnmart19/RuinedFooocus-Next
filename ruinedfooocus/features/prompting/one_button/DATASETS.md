# One Button data contract

The editable CSVs in `csvfiles/` are the authoritative bundled representation.
They retain the project's existing One Button Prompt attribution and license;
this change adds no external dataset or new license claim.

Logical identities are directory plus basename (without extension). In the CSV
API, `csvfiles/special_lists/tokenwriter` aliases `csvfiles/tokens`. Both contain
the same 590,381 ordered rows, starting with `10-point`; there is no token header.
The old `tokens` consumer deliberately still passes `skipheader=True`, omitting
that first word. Callers using `skipheader=False` retain it. The removed duplicate
payload's SHA-256 was
`1e5111905182f16541fec1331f4c49340f80e207d0812e5510877286e2ddb20c`;
the canonical payload's SHA-256 is
`879d2bedae6dbdf51ead550a54ebe97e104d6ba0ae94fdbc02911a5ddde964a9`.
The byte difference was its final CRLF, with identical CSV rows.

| API / identity                                              | Delimiter                   | Header and shape                                                                                 |
| ----------------------------------------------------------- | --------------------------- | ------------------------------------------------------------------------------------------------ |
| `csv_to_list`, ordinary datasets and custom CSV/TXT         | `;` unless caller overrides | Caller-owned `skipheader`; first column is value, second is gender when filtering                |
| `random_read_from_csv`                                      | `,`                         | No header skip; first column, duplicate rows keep selection weight                               |
| `artists_and_category`                                      | `,`                         | Named Artist, Tags, Medium, Description and category columns; rows and duplicates remain ordered |
| `special_lists/negativewords`                               | `;`                         | Named primer/negative columns; paired rows never filtered independently                          |
| `config/default_config[_suffix]` and user `config[_suffix]` | `;`                         | First row ignored as legacy header; comments start with `#` in any cell; two columns required    |

Encoding is UTF-8 with an optional BOM. Raw cached rows preserve all columns,
order, duplicates and empty physical rows. Projection skips empty rows. Bundled
CSVs retain the permissive legacy quote parser: for example card names contain
`"Princess" Yuna Moritani`. Custom CSVs use strict quoting and report malformed
quotes with file and line. Gender-short rows and short named-table/config rows
produce a source/row warning and are skipped. Missing optional files and empty
files yield no choices; missing required named columns fail with a diagnostic.

Ordinary value projection retains stable first-case-insensitive deduplication,
without trimming unless requested. Antilist matching uses lowercase, stripped
values. Matrix mode returns fresh nested lists and retains the historical nested
addon shape. A TXT sibling follows CSV; in matrix mode it replaces the CSV
projection, while scalar mode concatenates before stable deduplication.

User files resolve per filename in this order, without moving or modifying them:

1. `RF_DATA_DIR/random_prompt/userfiles` (or clone root when unset).
2. The legacy clone's `random_prompt/userfiles`.
3. Former package `one_button/userfiles`.

The same rule applies to direct custom subjects, artists, outfits, prefixes,
middles, suffixes, antilists and `userfiles/wildcards`. Packaged main, special-list
and template datasets accept `_replace.csv`. Main and special-list datasets
accept `_addon.csv`. For compatibility, addon names follow the selected name,
including `_replace`, `_light` or `_medium`. Templates do not accept addons.
Replacement wins over insanity variants. Variant existence checks use the variant
they select; RNG draws stay at the original call points, even after selecting a
light list. No final random choice or filtered request result is cached.

`data_store` lazily caches immutable tuples and immutable ordered artist indexes,
bounded to 192 entries and 96 MiB of recursively counted retained row/index payloads.
Shared objects count once per entry and conservatively again across entries;
this is not an RSS or transient-allocation limit. Identity includes absolute
path, nanosecond modification/creation times, size and inode plus parser options.
Every discovery resolves file existence anew, so create/delete/replacement changes
take effect. Caller lists and artist projections are fresh mutable copies.
Nothing loads token rows merely by importing modules or constructing the UI.

Config and preset discovery is read-only and falls back to bundled defaults.
Only explicit preset Save writes JSON. Developer token utilities require an
explicit destination outside the One Button package:
`export_token_dictionary(path)` copies all token rows as UTF-8 LF;
`sort_and_dedupe_csv_file(path)` writes a sorted lowercase filtered copy. Neither
can overwrite the shared bundled canonical source or its logical alias.

Plaintext plus lazy cache remains the storage policy. Standard-library gzip
reduces distribution bytes but adds decoding to cold parsing and gives no benefit
to warm in-memory reads; it is not a second authoritative shipped format.
