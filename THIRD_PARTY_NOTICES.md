# Third-party notices

VLAFlow includes derived code. Original source-file copyright and license headers are retained. Li Auto. claims copyright only in its contributions.

## Upstream project license (verbatim)

```text
MIT License

    Copyright (c) StarVLA Team.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    Rebases are allowed for forks and feature branches. When rebasing from upstream StarVLA, use descriptive commit messages, e.g., "chore: clone from StarVLA".
    Preserve attribution: keep at least the two latest upstream StarVLA commits as separate.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE
```

## Other components

| Component | Included code | License / notice |
| --- | --- | --- |
| NVIDIA | `vlaflow/dataloader/gr00t_lerobot/` and `vlaflow/model/modules/action_model/flow_matching_head/action_encoder.py` | Apache-2.0; retain per-file SPDX headers; [license](LICENSES/Apache-2.0.txt) |
| OpenVLA / Prismatic attribution | `vlaflow/training/trainer_utils/overwatch.py` | MIT; [OpenVLA license](LICENSES/OpenVLA.txt) |
| SimplerEnv | `examples/SimplerEnv/eval_files/evaluation.py` | MIT; [license](LICENSES/SimplerEnv.txt) |
| robosuite | Quaternion conversion in `examples/LIBERO/eval_files/eval_libero.py` and `examples/LIBERO-plus/eval_files/eval_libero.py` | MIT with the upstream accompanying notice; [license](LICENSES/robosuite.txt) |
| msgpack-numpy | `deployment/model_server/tools/msgpack_numpy.py` | BSD-3-Clause; [license](LICENSES/msgpack-numpy.txt) |
| Existing VLAFlow project website and figures | `index.html` and `assets/` | [Apache-2.0](LICENSES/Project-materials-Apache-2.0.txt); [original NOTICE](LICENSES/Project-materials-NOTICE.txt) |
| Updated VLAFlow technical report | `report/VLAFlow_Technical_Report.pdf` | Author-supplied arXiv:2607.01586v2 PDF; original embedded metadata retained; see below |

Source-file attributions and notices continue to apply. This inventory records identified
components; it is not a certification that all transitive provenance has been audited.
Separately installed dependencies, downloaded backbones, datasets, and simulators remain
subject to their own terms.

## Technical Report

On 2026-09-20, the previous report was replaced with the PDF supplied by the authors:
**arXiv:2607.01586v2**, dated August 4, 2026, with 38 PDF pages. Only the filename was
changed to preserve the website and README download paths; the PDF bytes were not edited.
Its embedded `/License` value is `http://arxiv.org/licenses/nonexclusive-distrib/1.0/`.
The code and website license statements must not be read as assigning Apache-2.0 or MIT
to this replacement PDF. Confirm any additional report reuse permissions with its authors.

## Upstream References

The missing license texts above were checked against these upstream sources on 2026-09-20:

- OpenVLA: <https://github.com/openvla/openvla/blob/main/LICENSE>
- robosuite, at the revision cited in the evaluation source:
  <https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/LICENSE>
- msgpack-numpy: <https://github.com/lebedov/msgpack-numpy/blob/master/LICENSE.md>
- StarVLA: <https://github.com/starVLA/starVLA/blob/starVLA_dev/LICENSE>

The retained StarVLA text contains commit-history wording in addition to standard MIT terms.
Do not delete this wording or assume that replacing the root license removes upstream
obligations. See [Release preparation](docs/releasing.md) for the unresolved review items.
