#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import concurrent.futures
import shutil
import time
import sys
import zipfile
from pathlib import Path
from typing import Any

import requests


PWD_ID = "927aef4f19a8"
PASSCODE = "S4SC"
TARGET = Path("/home/hajimi2025/CMS/quark_downloads/cxr_first.zip")
ORIGIN = "https://pan.quark.cn"
HOSTS = ("https://pan.quark.cn", "https://drive-pc.quark.cn", "https://drive-m.quark.cn")
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/131 Safari/537.36"


def fail(message: str) -> None:
    raise RuntimeError(message)


def api_json(session: requests.Session, method: str, url: str, **kwargs: Any) -> dict[str, Any]:
    response = session.request(method, url, timeout=60, **kwargs)
    try:
        data = response.json()
    except ValueError:
        fail(f"夸克接口返回非 JSON：HTTP {response.status_code} {response.text[:200]}")
    if not isinstance(data, dict):
        fail(f"夸克接口返回格式异常：{data!r}")
    return data


def find_items(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        for key in ("list", "items", "file_list"):
            candidate = value.get(key)
            if isinstance(candidate, list) and all(isinstance(item, dict) for item in candidate):
                return candidate
        for candidate in value.values():
            found = find_items(candidate)
            if found:
                return found
    return []


def get_detail(session: requests.Session, pdir_fid: str = "0") -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = {
        "pwd_id": PWD_ID,
        "passcode": PASSCODE,
        "pdir_fid": pdir_fid,
        "force": 0,
        "page": 1,
        "size": 100,
        "fetch_banner": 1 if pdir_fid == "0" else 0,
        "fetch_share": 1 if pdir_fid == "0" else 0,
        "fetch_relate_conversation": 1 if pdir_fid == "0" else 0,
        "fetch_total": 1,
        "fetch_sub_file_cnt": 1,
        "sort": "file_type:asc,file_name:asc",
        "support_visit_limit_private_share": True,
    }
    params = {"format": "png", "pr": "ucpro", "fr": "pc", "uc_param_str": ""}
    last: dict[str, Any] | None = None
    for host in HOSTS:
        data = api_json(session, "POST", f"{host}/1/clouddrive/share/sharepage/v2/detail", params=params, json=payload)
        last = data
        if data.get("status") == 200 and data.get("data"):
            return data, find_items(data["data"])
    fail(f"获取分享目录失败：{last}")


def locate_dir(session: requests.Session, fid: str, token: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    detail, items = get_detail(session, fid)
    current_token = detail.get("data", {}).get("token_info", token)
    for item in items:
        if item.get("file_name") == "cxr_first.zip":
            return item, current_token
    for item in items:
        if item.get("dir"):
            found, child_token = locate_dir(session, str(item.get("fid", "")), current_token)
            if found is not None:
                return found, child_token
    return None, current_token


def locate_file(session: requests.Session) -> tuple[dict[str, Any], dict[str, Any]]:
    root, items = get_detail(session)
    token = root.get("data", {}).get("token_info", {})
    for item in items:
        if item.get("file_name") == "cxr_first.zip":
            return item, token
    for item in items:
        if item.get("dir"):
            found, child_token = locate_dir(session, str(item.get("fid", "")), token)
            if found is not None:
                return found, child_token
    fail(f"分享目录未找到 cxr_first.zip，目录项：{[x.get('file_name') for x in items]}")


def get_download_url(session: requests.Session, item: dict[str, Any], token: dict[str, Any]) -> str:
    fid = str(item.get("fid") or "")
    stoken = str(token.get("stoken") or item.get("stoken") or "")
    if not fid or not stoken:
        fail(f"文件或分享令牌不完整：fid={fid!r}, token_keys={list(token)}")
    if int(item.get("size") or 0) > 50 * 1024**3:
        direct = api_json(
            session,
            "POST",
            "https://drive-pc.quark.cn/1/clouddrive/file/download",
            params={"pr": "ucpro", "fr": "pc", "sys": "win32", "ve": "2.5.56", "ut": "", "guid": ""},
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) quark-cloud-drive/2.5.56 Electron/18.3.5.12 Safari/537.36 Channel/pckk_other_ch"},
            json={"fids": [fid]},
        )
        direct_rows = direct.get("data") or []
        if direct_rows and direct_rows[0].get("download_url"):
            return str(direct_rows[0]["download_url"])
        return get_large_file_url(session, item, stoken)
    payload: dict[str, Any] = {"fids": [fid], "pwd_id": PWD_ID, "stoken": stoken}
    share_fid_token = item.get("share_fid_token")
    if share_fid_token:
        payload["fids_token"] = [share_fid_token]
    last: dict[str, Any] | None = None
    for host in HOSTS:
        data = api_json(session, "POST", f"{host}/1/clouddrive/file/share/download", json=payload)
        last = data
        rows = data.get("data") or []
        if isinstance(rows, list) and rows and rows[0].get("download_url"):
            return str(rows[0]["download_url"])
    fallback_payload = {
        "fids": [fid],
        "fids_token": [share_fid_token] if share_fid_token else [],
        "pwd_id": PWD_ID,
        "stoken": stoken,
    }
    for host in HOSTS:
        data = api_json(
            session,
            "POST",
            f"{host}/1/clouddrive/file/download",
            params={"entry": "ft", "fr": "pc", "pr": "ucpro"},
            json=fallback_payload,
        )
        last = data
        rows = data.get("data") or []
        if isinstance(rows, list) and rows and rows[0].get("download_url"):
            return str(rows[0]["download_url"])
    fail(f"获取下载地址失败：{last}")


def get_large_file_url(session: requests.Session, item: dict[str, Any], stoken: str) -> str:
    stamp = str(int(time.time()))
    folder_name = f"cms-quarkdl-{stamp}"
    headers = {"User-Agent": UA, "Accept": "application/json, text/plain, */*", "Content-Type": "application/json", "Origin": ORIGIN, "Referer": f"{ORIGIN}/"}
    params = {"pr": "ucpro", "fr": "pc", "uc_param_str": "", "__dt": "1000", "__t": str(int(time.time() * 1000))}
    created = api_json(session, "POST", "https://drive-pc.quark.cn/1/clouddrive/file", params=params, headers=headers, json={"pdir_fid": "0", "file_name": folder_name, "dir_path": "", "dir_init_lock": False})
    folder_fid = str((created.get("data") or {}).get("fid") or "")
    if not folder_fid:
        fail(f"创建临时夸克目录失败：{created}")
    fid = str(item.get("fid") or "")
    share_fid_token = str(item.get("share_fid_token") or "")
    saved = api_json(session, "POST", "https://drive.quark.cn/1/clouddrive/share/sharepage/save", params=params, headers=headers, json={"fid_list": [fid], "fid_token_list": [share_fid_token], "to_pdir_fid": folder_fid, "pwd_id": PWD_ID, "stoken": stoken, "pdir_fid": "0", "scene": "link"})
    task_id = str((saved.get("data") or {}).get("task_id") or "")
    if not task_id:
        fail(f"保存分享文件到临时目录失败：{saved}")
    for retry in range(300):
        time.sleep(1)
        task = api_json(session, "GET", "https://drive-pc.quark.cn/1/clouddrive/task", params={**params, "task_id": task_id, "retry_index": str(retry), "__t": str(int(time.time() * 1000))}, headers=headers)
        task_data = task.get("data") or {}
        if task_data.get("status") == 2:
            break
        if task.get("code") not in (0, None) and task.get("status") not in (200, None):
            fail(f"保存分享任务失败：{task}")
    else:
        fail("保存分享任务超时")
    rows = []
    for page in range(1, 5):
        listing = api_json(session, "GET", "https://drive-pc.quark.cn/1/clouddrive/file/sort", params={**params, "pdir_fid": folder_fid, "_page": str(page), "_size": "100", "_fetch_total": "1", "_fetch_sub_dirs": "1", "_sort": "file_type:asc,file_name:asc"}, headers=headers)
        batch = (listing.get("data") or {}).get("list") or []
        rows.extend(batch)
        if len(batch) < 100:
            break
    owned = next((row for row in rows if row.get("file_name") == item.get("file_name")), None)
    if not owned:
        fail(f"临时目录中未找到已保存文件：{[row.get('file_name') for row in rows]}")
    owned_fid = str(owned.get("fid") or "")
    info = api_json(session, "POST", "https://drive-pc.quark.cn/1/clouddrive/file/download", params={**params, "sys": "win32", "ve": "2.5.56", "ut": "", "guid": ""}, headers={**headers, "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) quark-cloud-drive/2.5.56 Electron/18.3.5.12 Safari/537.36 Channel/pckk_other_ch"}, json={"fids": [owned_fid]})
    rows = info.get("data") or []
    if not rows or not rows[0].get("download_url"):
        fail(f"获取临时文件下载地址失败：{info}")
    return str(rows[0]["download_url"])


def download(session: requests.Session, url: str, target: Path, expected_size: int | None) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if expected_size is None:
        fail("下载缺少预期大小")
    chunks_dir = target.with_suffix(target.suffix + ".chunks")
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunk_size = 16 * 1024 * 1024
    ranges = [(start, min(start + chunk_size, expected_size) - 1) for start in range(0, expected_size, chunk_size)]

    def fetch(spec: tuple[int, int, int]) -> int:
        index, start, end = spec
        chunk_path = chunks_dir / f"{index:06d}.part"
        want = end - start + 1
        if chunk_path.exists() and chunk_path.stat().st_size == want:
            return want
        for attempt in range(1, 6):
            try:
                headers = {"User-Agent": UA, "Origin": ORIGIN, "Referer": f"{ORIGIN}/", "Range": f"bytes={start}-{end}"}
                with session.get(url, headers=headers, stream=True, timeout=(60, 300)) as response:
                    if response.status_code != 206:
                        raise RuntimeError(f"HTTP {response.status_code}")
                    temp = chunk_path.with_suffix(".tmp")
                    count = 0
                    with temp.open("wb") as handle:
                        for block in response.iter_content(1024 * 1024):
                            if block:
                                handle.write(block)
                                count += len(block)
                    if count != want:
                        raise RuntimeError(f"chunk size {count}, expected {want}")
                    temp.replace(chunk_path)
                    return count
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(min(attempt * 2, 10))
        return 0

    all_specs = [(index, start, end) for index, (start, end) in enumerate(ranges)]
    done = sum((end - start + 1) for index, start, end in all_specs if (chunks_dir / f"{index:06d}.part").exists() and (chunks_dir / f"{index:06d}.part").stat().st_size == end - start + 1)
    specs = [(index, start, end) for index, start, end in all_specs if not ((chunks_dir / f"{index:06d}.part").exists() and (chunks_dir / f"{index:06d}.part").stat().st_size == end - start + 1)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(fetch, spec) for spec in specs]
        for future in concurrent.futures.as_completed(futures):
            done += future.result()
            print(f"\r下载 {done / expected_size * 100:6.2f}%", end="", flush=True)
    print()
    part = target.with_suffix(target.suffix + ".part")
    with part.open("wb") as output:
        for index, (start, end) in enumerate(ranges):
            chunk_path = chunks_dir / f"{index:06d}.part"
            if chunk_path.stat().st_size != end - start + 1:
                fail(f"分片校验失败：{index}")
            with chunk_path.open("rb") as source:
                shutil.copyfileobj(source, output, 1024 * 1024)
    if part.stat().st_size != expected_size:
        fail(f"文件大小校验失败：实际 {part.stat().st_size}，预期 {expected_size}")
    part.replace(target)
    shutil.rmtree(chunks_dir)


def main() -> int:
    cookie = sys.stdin.read().strip()
    if "=" not in cookie or "__uid=" not in cookie:
        fail("标准输入必须是包含 __uid=... 的完整 Cookie 行")
    session = requests.Session()
    session.trust_env = False
    session.headers.update({"User-Agent": UA, "Accept": "application/json, text/plain, */*", "Origin": ORIGIN, "Referer": f"{ORIGIN}/", "Cookie": cookie})
    account = api_json(session, "GET", f"{ORIGIN}/account/info", params={"fr": "pc", "platform": "pc"})
    if not account.get("data"):
        fail(f"Cookie 登录验证失败：status={account.get('status')} code={account.get('code')} message={account.get('message')}")
    print(f"登录账号：{account['data'].get('nickname', '<unknown>')}")
    item, token = locate_file(session)
    print(f"定位文件：{item.get('file_name')}，大小 {item.get('size')} bytes")
    url = get_download_url(session, item, token)
    download(session, url, TARGET, int(item.get("size") or 0) or None)
    digest = hashlib.sha256(TARGET.read_bytes()).hexdigest()
    with zipfile.ZipFile(TARGET) as archive:
        bad = archive.testzip()
        if bad:
            fail(f"ZIP 校验失败，首个损坏成员：{bad}")
        count = len(archive.infolist())
    print(json.dumps({"path": str(TARGET), "size": TARGET.stat().st_size, "sha256": digest, "zip_members": count}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(1)
