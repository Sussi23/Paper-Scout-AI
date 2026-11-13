# PaperScout AI — Backend

Một AI agent thật (Gemini function calling qua thư viện `google-genai`) tìm kiếm bài báo khoa học trên cơ sở dữ liệu
OpenAlex và sàng lọc mức độ liên quan, được cung cấp dưới dạng dịch vụ FastAPI cho website gọi. Thư mục này **chỉ chứa backend**.

- **Gemini quyết định**: gọi tool nào, dùng từ khoá gì, đổi truy vấn thế nào sau khi xem kết quả, và khi nào dừng.
- **Code quyết định**: thực thi tool, kiểm tra tham số, ngân sách tìm kiếm, nguồn sự thật của metadata, xử lý lỗi,
  tính trạng thái cuối và ghi log.
- Tên bài, tác giả, năm, DOI, abstract và URL **chỉ lấy từ OpenAlex** (lưu trong registry theo `paper_id`).
  Gemini chỉ trả về `paper_id` + nhãn + lý do.

Mọi lệnh dưới đây chạy từ **thư mục gốc của repo** (thư mục chứa `backend/`), không chạy từ trong `backend/`.

<!-- BẮT ĐẦU: DÀNH CHO TV5 (dán vào README gốc) -->
## Cài đặt (Installation)

Cần Python 3.10 trở lên (đã kiểm tra trên 3.10.0).

```bash
python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1      macOS/Linux:  source .venv/bin/activate
pip install -r backend/requirements.txt
cp .env.example .env        # Windows PowerShell: Copy-Item .env.example .env
```

Điền file `.env` (không bao giờ commit file này): `GEMINI_API_KEY`, `OPENALEX_API_KEY` và `GEMINI_MODEL`.
Cố ý không có model mặc định; chạy `python scripts/list_models.py` để xem key của bạn dùng được model nào.

## Cách chạy (How to Run)

```bash
uvicorn backend.main:app --reload      # API tại http://127.0.0.1:8000, tài liệu tương tác tại /docs
```

Kiểm tra bằng `GET /health`. Một lần chạy nghiên cứu mất khoảng 1–2 phút, nên phía client phải chờ ít nhất 3 phút.

## Kiểm thử (Testing)

```bash
pip install -r backend/requirements-dev.txt
pytest                                  # chạy offline: không cần mạng, không cần key
```

Test live gọi API thật của Gemini/OpenAlex (tốn quota) mặc định bị tắt; xem mục "3. Kiểm thử" bên dưới.
<!-- KẾT THÚC: DÀNH CHO TV5 -->

## 1. Cấu hình

Đọc từ biến môi trường, hoặc từ file `.env` ở thư mục gốc repo (biến môi trường được ưu tiên hơn).
Server vẫn khởi động được khi chưa có key (để `/health` hoạt động), nhưng `POST /api/research` trả 503 cho đến khi
đặt đủ các biến bắt buộc. File `.env.example` liệt kê đầy đủ tên các biến.

| Biến | Bắt buộc | Mặc định | Ý nghĩa |
|---|---|---|---|
| `GEMINI_API_KEY` | có | không | API key của Google AI Studio |
| `GEMINI_MODEL` | có | không | Tên model Gemini (xem `scripts/list_models.py`) |
| `GEMINI_TEMPERATURE` | không | mặc định của model | Để trống trừ khi bạn đang thử nghiệm |
| `GEMINI_MIN_INTERVAL_SECONDS` | không | `2` | Khoảng cách tối thiểu giữa các lần gọi Gemini (giúp tránh lỗi 429) |
| `OPENALEX_API_KEY` | có | không | API key OpenAlex, gửi qua header `Authorization: Bearer` |
| `MAX_SEARCHES` | không | `3` | Ngân sách tìm kiếm mỗi lần chạy; `search_papers` và `find_related_papers` đều tính vào |
| `MAX_MODEL_STEPS` | không | `12` | Số lượt model tối đa mỗi lần chạy |
| `SEARCH_PAGE_SIZE` | không | `25` | Số kết quả mỗi lần tìm (đồng thời là mức trần của tham số `limit` trong tool) |
| `ABSTRACT_CHARS_FOR_MODEL` | không | `1200` | Số ký tự abstract đưa cho model |
| `MAX_CONCURRENT_RUNS` | không | `1` | Số lần chạy song song; lần chạy khác chờ tối đa 90 giây rồi nhận HTTP 429 |
| `ALLOWED_ORIGINS` | không | rỗng | Danh sách origin của frontend được CORS cho phép, phân tách bằng dấu phẩy |
| `LOG_DIR` | không | `logs` | Nơi ghi log JSONL của từng lần chạy |
| `RUNS_DIR` | không | `runs` | Nơi lưu `result.json` và `results.csv` |
| `CRITERIA_PATH` | không | `evaluation/screening_criteria.md` | Tiêu chí sàng lọc, được đọc lại ở mỗi lần chạy |

Key không bao giờ bị in ra: chúng bị che (`***`) trong log, thông báo lỗi, `tool_history` và kết quả API;
`/health` chỉ báo mỗi key đã được cấu hình hay chưa.

## 2. Hợp đồng API

URL gốc khi phát triển: `http://127.0.0.1:8000`. Tài liệu tương tác (Swagger UI): `/docs`.

### `POST /api/research`

Chạy agent đồng bộ (khoảng 1–2 phút) rồi trả kết quả.

Nội dung request:

| Trường | Kiểu | Quy tắc |
|---|---|---|
| `topic` | string | bắt buộc, 3–300 ký tự sau khi cắt khoảng trắng hai đầu |
| `year_from`, `year_to` | số nguyên hoặc null | từ 1900 đến năm sau năm hiện tại; `year_from <= year_to` |
| `num_papers` | số nguyên | 1–30, mặc định 10 |
| `criteria` | string hoặc null | tiêu chí sàng lọc bổ sung, tối đa 1000 ký tự |
| `language` | `"vi"` hoặc `"en"` | ngôn ngữ phần `reason` của từng bài, mặc định `"vi"` |

```json
{ "topic": "machine learning for medical image diagnosis", "num_papers": 8, "language": "en" }
```

Response (trích từ [`demo/sample_result.json`](../demo/sample_result.json), một lần chạy thật; các giá trị dài đã rút gọn):

```json
{
  "run_id": "e29b2bbd33a4",
  "status": "completed",
  "message": "Found 16/8 Relevant papers after 1 searches.",
  "selected_ids": ["W3169509062", "W3173991394", "..."],
  "selected": [
    {
      "metadata": {
        "paper_id": "W3169509062",
        "title": "Machine learning for medical imaging-based COVID-19 detection and diagnosis",
        "authors": ["Rokaya Rehouma", "Michael Büchert"],
        "year": 2021,
        "doi": "https://doi.org/10.1002/int.22504",
        "source_url": "https://doi.org/10.1002/int.22504",
        "abstract": "The novel coronavirus disease 2019 (COVID-19) is considered to be ...",
        "source": "openalex"
      },
      "label": "Relevant",
      "reason": "The paper focuses on machine learning applications using medical imaging ..."
    }
  ],
  "label_counts": { "Relevant": 16, "Possibly Relevant": 1, "Not Relevant": 0, "Insufficient Information": 8 },
  "searches_used": 1,
  "queries": ["machine learning medical image diagnosis"],
  "tool_history": [
    {
      "step": 1, "tool": "search_papers",
      "args": { "query": "machine learning medical image diagnosis", "rationale": "..." },
      "rationale": "Initial search to find papers focusing on ...",
      "status": "ok", "error_code": null,
      "summary": "search_papers('machine learning medical image diagnosis') -> 25 trả về, 25 mới",
      "latency_ms": 1873, "timestamp": "2026-11-09T16:45:17.165387Z"
    }
  ],
  "errors": [],
  "terminated_by": "model_finish",
  "model": "gemini-3.5-flash",
  "started_at": "2026-11-09T16:45:10.953374Z",
  "finished_at": "2026-11-09T16:45:35.185462Z"
}
```

(File mẫu được ghi trước khi đổi cách diễn đạt trường `message` cho trường hợp vượt mục tiêu; hiện nay một lần chạy như
vậy sẽ ghi "Đã tìm đủ: 16 bài Relevant (mục tiêu 8) ...". Không nên phân tích `message` bằng code; hãy dùng `status`.)

`status` do code tính, không phải do model:

| `status` | Ý nghĩa |
|---|---|
| `completed` | số bài Relevant >= `num_papers` |
| `partial` | ít bài Relevant hơn yêu cầu nhưng ít nhất một lượt tìm thành công (`message` nêu số thực tế `X/Y`) |
| `error` | không lượt tìm nào thành công, hoặc Gemini lỗi không phục hồi được (xem `errors`, `terminated_by`) |

**Cả ba trạng thái `completed`, `partial`, `error` đều trả HTTP 200; trạng thái luôn nằm trong JSON.**
`selected` gồm toàn bộ bài Relevant, sau đó đến Possibly Relevant, theo thứ tự phát hiện, cắt ở `num_papers`.
`label_counts` đếm mọi bài đã gắn nhãn, kể cả Not Relevant và Insufficient Information.
`terminated_by` là một trong `model_finish`, `model_text_stop`, `step_cap`, `budget`, `llm_error`.

Các mã HTTP khác: `422` input sai (`{"detail": "Invalid request: ...", "errors": [{"field", "message"}]}`),
`429` server đang bận chạy yêu cầu khác (đã chờ 90 giây), `503` server thiếu cấu hình bắt buộc (thông báo nêu tên
biến còn thiếu), `500` lỗi không lường trước (không lộ chi tiết nội bộ).

**Timeout:** một lần chạy thường mất 1–2 phút (Gemini có thể mất 15–30 giây để sàng lọc 25 bài). Frontend phải đặt
timeout phía client ít nhất 3 phút và hiển thị trạng thái chờ.

### Trường bổ sung cho giao diện: `metrics`, `screened_papers`, `funnel`

Ba trường này được **thêm sau** để giao diện thể hiện rõ hoạt động của agent. Không trường cũ nào bị đổi tên, xoá hay
đổi kiểu. Kết quả lưu trước khi có các trường này (ví dụ `demo/sample_result.json`) **không có** chúng: khi đọc lại,
`metrics` là `null`, `screened_papers` là `[]`, `funnel` là `null`, nên giao diện phải chấp nhận các giá trị đó.
Mọi con số đều do code tính, không gửi cho model và không lấy từ model.

**`tool_history[].metrics`** (đối tượng `StepMetrics`, có ở mọi bước của kết quả mới):

| Trường | Có khi nào | Ý nghĩa |
|---|---|---|
| `returned`, `new_papers`, `dropped_year`, `dropped_duplicate`, `dropped_invalid` | chỉ bước `search_papers` / `find_related_papers` thành công (bước khác: `null`) | số bản ghi OpenAlex trả về, số bài mới, số bị loại vì năm, vì trùng, vì thiếu dữ liệu |
| `accepted`, `rejected`, `relabeled` | chỉ bước `record_screening` | số mục được chấp nhận, bị từ chối, được gắn nhãn lại |
| `labels_this_step` | chỉ bước `record_screening` | số mục được chấp nhận theo từng nhãn trong lần gọi đó (đủ 4 khoá) |
| `papers_found`, `screened`, `unscreened`, `relevant`, `possibly_relevant`, `searches_used` | **mọi bước**, kể cả bước bị từ chối hoặc lỗi | ảnh chụp trạng thái **sau** bước đó |

**`screened_papers`**: mọi bài agent đã tìm thấy, theo thứ tự phát hiện (khác `selected` chỉ gồm bài được chọn).

| Trường | Ý nghĩa |
|---|---|
| `paper_id`, `title`, `authors`, `year`, `doi`, `source_url` | lấy nguyên văn từ OpenAlex (không lấy từ model) |
| `has_abstract` | bài có abstract hay không (abstract không đưa vào đây để kết quả nhẹ; xem `selected` hoặc CSV) |
| `label`, `reason` | nhãn và lý do model gán; `null` nếu bài chưa được sàng lọc |
| `selected` | `true` nếu `paper_id` có trong `selected_ids` |
| `found_at_step`, `found_by_tool`, `found_via` | bước, công cụ và truy vấn đã tìm ra bài lần đầu; `found_via` là chuỗi truy vấn của `search_papers`, hoặc `related:<seed_id>` với `find_related_papers` |

**`funnel`**: phễu tổng hợp qua các lượt tìm thành công: `returned`, `dropped_invalid`, `dropped_year`,
`dropped_duplicate`, `unique_found`, `screened`, `selected`. Luôn đúng:
`returned == dropped_invalid + dropped_year + dropped_duplicate + unique_found`.

Ví dụ **minh hoạ** (dữ liệu giả từ fixture test, KHÔNG phải kết quả thật; đã rút gọn):

```json
{
  "status": "completed",
  "tool_history": [
    { "step": 1, "tool": "search_papers", "status": "ok",
      "metrics": { "returned": 8, "new_papers": 5, "dropped_year": 2, "dropped_duplicate": 0, "dropped_invalid": 1,
                   "accepted": null, "rejected": null, "relabeled": null, "labels_this_step": null,
                   "papers_found": 5, "screened": 0, "unscreened": 5, "relevant": 0, "possibly_relevant": 0,
                   "searches_used": 1 } },
    { "step": 2, "tool": "record_screening", "status": "ok",
      "metrics": { "returned": null, "new_papers": null, "dropped_year": null, "dropped_duplicate": null,
                   "dropped_invalid": null, "accepted": 5, "rejected": 0, "relabeled": 0,
                   "labels_this_step": { "Relevant": 2, "Possibly Relevant": 1, "Not Relevant": 1,
                                         "Insufficient Information": 1 },
                   "papers_found": 5, "screened": 5, "unscreened": 0, "relevant": 2, "possibly_relevant": 1,
                   "searches_used": 1 } }
  ],
  "screened_papers": [
    { "paper_id": "W9000000001", "title": "Synthetic Study One on Adaptive Learning Systems",
      "authors": ["Alice Example", "Bob Sample"], "year": 2021, "doi": "https://doi.org/10.5555/syn.0001",
      "source_url": "https://doi.org/10.5555/syn.0001", "has_abstract": true, "label": "Relevant",
      "reason": "Evaluates adaptive learning systems in schools.", "selected": true,
      "found_at_step": 1, "found_by_tool": "search_papers", "found_via": "adaptive learning" },
    { "paper_id": "W9000000004", "title": "Synthetic Study Four Without Abstract", "authors": ["Eve Placeholder"],
      "year": 2023, "doi": null, "source_url": "https://example.org/syn/0004", "has_abstract": false,
      "label": "Insufficient Information", "reason": "No abstract available.", "selected": false,
      "found_at_step": 1, "found_by_tool": "search_papers", "found_via": "adaptive learning" }
  ],
  "funnel": { "returned": 8, "dropped_invalid": 1, "dropped_year": 2, "dropped_duplicate": 0,
              "unique_found": 5, "screened": 5, "selected": 2 }
}
```

Lưu ý cho giao diện: `tool_history[].args` của `record_screening` chứa toàn bộ danh sách đánh giá model gửi (có thể
rất dài, hàng chục mục), nên **không nên hiển thị thô**; hãy dùng `metrics` và `screened_papers` thay thế.
Schema đầy đủ của các trường này hiển thị tại `/docs` (mục `ResearchResult`, `StepMetrics`, `ScreenedPaper`, `Funnel`).

### `GET /api/runs/{run_id}`

Trả lại `result.json` đã lưu của lần chạy, gồm cả `tool_history` cho màn hình Agent Activity. `404` nếu không tồn tại.
Endpoint này trả nguyên file JSON đã lưu, nên kết quả cũ có thể thiếu `screened_papers`, `funnel` và `metrics`.

### `GET /api/exports/{run_id}`

Trả file CSV của `selected` (`text/csv`, UTF-8 có BOM để Excel đọc đúng tiếng Việt, tên file
`paperscout_{run_id}.csv`; các cột Title, Authors, Year, DOI, Source URL, Abstract, Relevance, Relevance Reason).
`404` nếu không tồn tại. `run_id` phải khớp `^[a-f0-9]{12}$`; giá trị khác bị từ chối (`422`, hoặc `404` khi giá trị
chứa dấu phân cách đường dẫn nên không khớp route), nhờ đó chống được path traversal.
File CSV tương ứng của lần chạy mẫu nằm ở [`demo/sample_results.csv`](../demo/sample_results.csv).

### `GET /health`

`{"status": "ok", "model": "...", "gemini_key_configured": true, "openalex_key_configured": true}`.

## 3. Kiểm thử

```bash
pytest                       # mặc định: offline, dùng FakeLLM / FakeOpenAlex và fixture tổng hợp
```

Test live (`tests/test_live_smoke.py`) gọi API thật và tốn quota; cần key thật trong `.env`:

```powershell
# Windows PowerShell
$env:RUN_LIVE = "1"; pytest -m live
```
```bash
# bash / zsh
RUN_LIVE=1 pytest -m live
```

Các file test: `test_agent.py` (vòng lặp agent, state, tool registry, run logger, client Gemini), `test_tools.py`
(client OpenAlex, tìm kiếm, xử lý dữ liệu, xuất CSV), `test_api.py` (FastAPI). Các file nhỏ khác: `test_config.py`,
`test_schemas.py`, `test_structure.py`, `test_baseline.py`, `test_show_log.py`, `test_scan_secrets.py`,
`test_live_smoke.py`. Đồ giả và dữ liệu dùng chung: `conftest.py`, `fakes.py`, `fixtures/`.
Báo cáo kiểm thử `tests/test_report.md` do thành viên phụ trách bổ sung.

## 4. Cấu trúc repo

Phần của backend (đã có đầy đủ) và các chỗ để trống cho thành viên khác:

```
backend/               (backend: đã có)
  main.py              ứng dụng FastAPI (create_app, dependencies)
  config.py            cấu hình đọc từ .env / biến môi trường, có kiểm tra hợp lệ
  requirements.txt     thư viện chạy, ghim phiên bản     requirements-dev.txt   thêm pytest
  README.md            tài liệu này
  agent/
    research_agent.py  run_research(): vòng lặp agent và kết quả cuối
    instructions.py    system instruction, task message, nạp tiêu chí sàng lọc
    schemas.py         các model Pydantic (request, metadata, đánh giá, kết quả)
    state.py           trạng thái mỗi lần chạy: registry bài báo, nhãn, ngân sách
    llm.py             client Gemini (giãn cách gọi, retry/backoff), giao diện LLMClient
    tool_registry.py   khai báo 4 tool, kiểm tra tham số, điều phối, mã lỗi
    run_logger.py      logger JSONL có che secret
  tools/
    search.py          client OpenAlex và các thao tác tìm kiếm / bài liên quan
    processing.py      chuẩn hoá, lọc theo năm, loại trùng
    export.py          xuất CSV
scripts/               các script hỗ trợ, chạy từ thư mục gốc repo (xem bảng bên dưới)
tests/                 bộ test offline (test_report.md: để trống, thành viên khác thêm)
demo/
  logs/                log của 3 lần demo thật
  sample_result.json   kết quả mẫu (JSON)
  sample_results.csv   kết quả mẫu (CSV)
  screenshots/         để trống: ảnh chụp màn hình, thành viên khác thêm
  (demo_video_link.md: thành viên khác thêm)
evaluation/            để trống: research_queries.csv, evaluation_results.csv, screening_criteria.md
                       (do thành viên phụ trách đánh giá thêm; bản nháp tiêu chí chưa đưa vào repo)
frontend/              để trống: giao diện website, thành viên khác thêm
docs/                  để trống: tài liệu, thành viên khác thêm
.env.example  .gitignore  pytest.ini
(README.md ở gốc repo: để trống, thành viên khác thêm)
```

Các script:

| Script | Mục đích | Tốn quota API |
|---|---|---|
| `scripts/list_models.py` | liệt kê model Gemini hỗ trợ `generateContent` | gần như không |
| `scripts/check_connection.py` | kiểm tra nhanh Gemini (văn bản + function calling) | 2 request nhỏ |
| `scripts/check_openalex.py` | kiểm tra nhanh OpenAlex (tìm kiếm + bài liên quan) | 1 lượt tìm |
| `scripts/live_demo.py` | chạy 3 request thật; ghi `demo/logs/` và `demo/results/` | có |
| `scripts/show_log.py` | in timeline dễ đọc của một file log | không |
| `scripts/baseline_script.py` | bản đối chứng không phải agent (từ khoá cố định, luôn 3 lượt tìm) | có |
| `scripts/scan_secrets.py` | quét file/lịch sử git tìm secret bị lộ (không in giá trị) | không |
| `scripts/check_env_example.py` | so tên biến trong `config.py` với `.env.example` | không |
| `scripts/verify_clean_clone.py` | clone repo vào thư mục tạm, cài đặt, chạy test và thử khởi động server | không |

`python scripts/show_log.py demo/logs/e29b2bbd33a4.jsonl` hiển thị timeline của lần chạy mẫu.

## 5. Vòng lặp agent hoạt động thế nào

```
state = RunState(request); contents = [task message]
for step in 1..MAX_MODEL_STEPS:
    response = gemini.generate(contents, tools, system_instruction)     # giãn cách gọi, retry khi 429/503
    contents.append(response.content)       # giữ NGUYÊN đối tượng Content của model (giữ thought signature)
    for each function call (cho phép gọi song song, xử lý lần lượt):
        result = registry.dispatch(call, state)       # kiểm tra -> thực thi -> bắt lỗi -> ghi log
    contents.append(user Content: mỗi lời gọi một function_response, cùng thứ tự)
    dừng khi finish_research được chấp nhận, hoặc hết ngân sách và mọi bài đã được sàng lọc
result = build_result(state)      # trạng thái và metadata do code tính từ registry
```

Tool (gọi function thủ công; tắt automatic function calling, chế độ AUTO). Mọi tool đều có tham số `rationale`
(vì sao chọn bước này, dựa trên kết quả trước), được ghi log và hiển thị trong `tool_history`.

| Tool | Chi phí | Ghi chú |
|---|---|---|
| `search_papers(query, rationale, year_from?, year_to?, limit?)` | 1 lượt tìm | năm bị kẹp về khoảng của request; chỉ trả về bài chưa thấy |
| `find_related_papers(seed_paper_id, rationale, limit?)` | 1 lượt tìm | bài hạt giống phải đang được gắn Relevant hoặc Possibly Relevant; dùng `related_works` của OpenAlex |
| `record_screening(assessments, rationale)` | miễn phí | kiểm tra từng mục; mục sai bị từ chối, mục đúng vẫn được ghi; cho phép gắn nhãn lại |
| `finish_research(reason)` | miễn phí | bị từ chối khi còn bài chưa sàng lọc; lần kết thúc sớm đầu tiên phải xác nhận |

Mã lỗi trả cho model: `SEARCH_BUDGET_EXHAUSTED`, `UNKNOWN_TOOL`, `INVALID_ARGS`, `EMPTY_QUERY`,
`UNKNOWN_PAPER_ID`, `INVALID_LABEL`, `SEED_NOT_ELIGIBLE`, `UNSCREENED_PAPERS_REMAIN`, `PREMATURE_FINISH_CONFIRM`,
`OPENALEX_AUTH`, `OPENALEX_RATE_LIMIT`, `OPENALEX_HTTP`, `OPENALEX_TIMEOUT`, và `INTERNAL_ERROR` cho exception
bất ngờ bên trong tool.

Tên bài và abstract được coi là dữ liệu không đáng tin; system instruction dặn model bỏ qua mọi chỉ dẫn nằm trong đó.
Tiêu chí sàng lọc được đọc từ `CRITERIA_PATH` ở mỗi lần chạy (đổi file là đổi hành vi); nếu thiếu file thì dùng bản
tiêu chí dựng sẵn và ghi cảnh báo vào log.

Mỗi lần chạy ghi `logs/{run_id}.jsonl`, mỗi dòng một sự kiện: `run_started`, `model_request`, `model_response`,
`tool_call`, `tool_result`, `guard` (từ chối do ngân sách, tool lạ, tham số sai, id bịa, kết thúc sớm, năm bị kẹp),
`run_finished`.

## 6. Triển khai trên Render

- Lệnh build: `pip install -r backend/requirements.txt`
- Lệnh khởi động: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
- `PYTHON_VERSION`: đặt một bản 3.10.x (phiên bản backend này đã được kiểm tra; chưa thử trên 3.11).
- Đường dẫn health check: `/health`
- Biến môi trường cần khai báo trên Render: `GEMINI_API_KEY`, `GEMINI_MODEL`, `OPENALEX_API_KEY`,
  `ALLOWED_ORIGINS` (origin của frontend đã deploy, ví dụ `https://ten-trang-web.example`); tuỳ chọn:
  `GEMINI_TEMPERATURE`, `GEMINI_MIN_INTERVAL_SECONDS`, `MAX_SEARCHES`, `MAX_MODEL_STEPS`, `SEARCH_PAGE_SIZE`,
  `ABSTRACT_CHARS_FOR_MODEL`, `MAX_CONCURRENT_RUNS`, `LOG_DIR`, `RUNS_DIR`, `CRITERIA_PATH`.
- Đĩa là tạm thời: `runs/` và `logs/` mất khi khởi động lại hoặc deploy lại, nên `GET /api/runs/{id}` và
  `GET /api/exports/{id}` chỉ dùng được cho các lần chạy gần đây. Máy chủ gói miễn phí cũng ngủ khi nhàn rỗi
  (request đầu tiên sau khoảng nghỉ sẽ chậm).

## 7. Giới hạn đã biết

- Chỉ mới kiểm tra với Python 3.10.0.
- Một lần chạy mất khoảng 1–2 phút; endpoint chạy đồng bộ, và với `MAX_CONCURRENT_RUNS=1` người dùng thứ hai phải chờ
  (tối đa 90 giây) hoặc nhận HTTP 429.
- Nhãn mức độ liên quan là phán đoán của Gemini dựa trên tên bài và abstract. Chưa có bộ đánh giá đối chiếu nên chất
  lượng nhãn chưa được đo.
- Quy tắc "bài không có abstract thì không bao giờ được gắn Relevant" chỉ nằm trong prompt và tiêu chí sàng lọc,
  code chưa ép cứng. (Trong 3 lần demo không có bài Relevant nào thiếu abstract.)
- Quota gói miễn phí: giới hạn của Gemini phụ thuộc model và key. Khi gặp 429/503, client chờ 4, 8, 16 giây và thử lại
  tối đa 3 lần, sau đó lần chạy kết thúc với `status: "error"`; key OpenAlex miễn phí có ngân sách nhỏ mỗi ngày
  (khoảng 1 USD/ngày; một lượt tìm từ khoá tốn khoảng 0,001 USD). Khi chạy demo nhiều lần trong một ngày, Gemini đã
  trả 503 (quá tải) rồi 429 (hết quota).
- `scripts/baseline_script.py` (bản đối chứng không phải agent) đã có test offline và đã thử chạy với API thật, nhưng
  bị Gemini ngắt giữa chừng (503, sau đó 429) nên **chưa có kết quả baseline hoàn chỉnh** trong `demo/`.
  Script không lưu kết quả của lần chạy bị ngắt; hãy chạy lại khi quota đã hồi.
- `demo/sample_result.json` và `demo/logs/` được tạo trước khi có `metrics`, `screened_papers`, `funnel`, nên không
  chứa các trường này; cần chạy lại một yêu cầu thật để có mẫu đầy đủ.
- `demo/logs/` chứa log 3 lần chạy demo thật: chủ đề phổ biến (`e29b2bbd33a4`), chủ đề hẹp (`398069570191`) và
  khoảng năm chặt (`f06d72185141`).
- `evaluation/screening_criteria.md` là bản nháp do thành viên phụ trách đánh giá quản lý và không nằm trong bản repo
  này; khi thiếu file, backend dùng tiêu chí dựng sẵn.

## 8. Nguồn tham khảo

Code được viết từ đầu dựa trên tài liệu hiện hành của Gemini function calling và của OpenAlex API (xác thực bằng API key,
`search`, `select`, `related_works`). Repo tham khảo `academic-online-research-agent` không được sử dụng và không có
đoạn code nào được chép từ đó.
