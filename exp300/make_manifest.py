"""AI Hub 경구약제 테스트셋(300장) -> manifest.csv 만들기 + 개발용/평가용 나누기.

폴더 구조(zip을 풀면):  <root>/images/*.png  +  <root>/labels/*.json  (이름 같은 쌍)
zip이 여러 개면 --roots 에 폴더를 모두 적는다.

실행 예) python make_manifest.py --roots 약제각인인식_테스트셋 "약제각인인식_테스트셋(1)" --out manifest.csv

정답(truth): 앞면 사진이면 print_front, 뒷면 사진이면 print_back.
정규화(truth_norm): 유니코드 정규화 -> 대문자 -> 공백 제거. 팀이 채점 규칙을 정하면 NORMALIZE 만 맞춰 바꾼다.
분할: 약제(drug_N)마다 dev_ratio 만큼을 개발용(dev), 나머지를 평가용(test)으로 나눈다(난수 시드 고정).
"""
import argparse, csv, json, random, re, sys, unicodedata
from collections import Counter, defaultdict
from pathlib import Path

FIELDS = ["file_name", "image_path", "drug_N", "dl_name", "form", "shape", "color1", "color2",
          "drug_dir", "back_color", "light_color", "camera_la", "camera_lo", "rot_deg", "size",
          "print_front", "print_back", "truth", "truth_norm", "has_mark", "split"]


def NORMALIZE(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    return re.sub(r"\s+", "", s).upper()


def load_label(p: Path):
    d = json.loads(p.read_text(encoding="utf-8"))
    im = d.get("images", d)
    if isinstance(im, list):
        im = im[0]
    return im


def rot_from_name(name: str):
    # 예) K-013392_0_2_0_2_90_000_200.png -> 끝에서 둘째 = 알약 회전각(000), 셋째 = 카메라각(90)
    parts = Path(name).stem.split("_")
    try:
        return int(parts[-2])
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--out", default="manifest.csv")
    ap.add_argument("--dev-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    rows, seen, missing_img, empty_truth = [], {}, [], []
    for root in map(Path, a.roots):
        labs = sorted((root / "labels").glob("*.json"))
        if not labs:
            sys.exit(f"[오류] {root}/labels 에 json 이 없어요. 폴더 이름/위치를 확인하세요.")
        for lp in labs:
            im = load_label(lp)
            fname = im.get("file_name") or im.get("imgfile") or lp.stem + ".png"
            ip = root / "images" / fname
            if not ip.exists():
                alt = list((root / "images").glob(lp.stem + ".*"))
                if alt:
                    ip = alt[0]
                else:
                    missing_img.append(lp.name); continue
            if fname in seen:
                print(f"[경고] 파일 이름 중복: {fname} ({seen[fname]} / {root})"); continue
            seen[fname] = str(root)
            front = (im.get("drug_dir") == "앞면")
            pf, pb = im.get("print_front", "") or "", im.get("print_back", "") or ""
            truth = pf if front else pb
            has_mark = bool(im.get("mark_code_front_anal") or im.get("mark_code_back_anal"))
            if not truth.strip():
                empty_truth.append(fname)
            rows.append(dict(file_name=fname, image_path=str(ip), drug_N=im.get("drug_N", ""),
                             dl_name=im.get("dl_name", ""), form=im.get("dl_custom_shape", ""),
                             shape=im.get("drug_shape", ""), color1=im.get("color_class1", ""),
                             color2=im.get("color_class2", ""), drug_dir=im.get("drug_dir", ""),
                             back_color=im.get("back_color", ""), light_color=im.get("light_color", ""),
                             camera_la=im.get("camera_la", ""), camera_lo=im.get("camera_lo", ""),
                             rot_deg=rot_from_name(fname), size=im.get("size", ""),
                             print_front=pf, print_back=pb, truth=truth, truth_norm=NORMALIZE(truth),
                             has_mark=int(has_mark), split=""))

    # 약제별로 섞어서 dev/test 나누기 (모든 약제가 양쪽에 들어가도록)
    rng = random.Random(a.seed)
    by_drug = defaultdict(list)
    for r in rows:
        by_drug[r["drug_N"]].append(r)
    for drug, rs in sorted(by_drug.items()):
        rng.shuffle(rs)
        n_dev = max(1, round(len(rs) * a.dev_ratio)) if len(rs) > 1 else 0
        for i, r in enumerate(rs):
            r["split"] = "dev" if i < n_dev else "test"

    rows.sort(key=lambda r: r["file_name"])
    with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)

    # ---- 점검 요약 ----
    print(f"\n이미지+라벨 쌍: {len(rows)}장  (이미지 없는 라벨 {len(missing_img)}개, 정답 각인이 빈 것 {len(empty_truth)}장)")
    for title, key in [("제형", "form"), ("약제", "drug_N"), ("앞/뒷면", "drug_dir"),
                       ("카메라각도", "camera_la"), ("알약회전(도)", "rot_deg"), ("모양", "shape"),
                       ("분할", "split"), ("마크 있음", "has_mark")]:
        c = Counter(r[key] for r in rows)
        print(f"- {title}: " + ", ".join(f"{k}={v}" for k, v in sorted(c.items(), key=lambda x: str(x[0]))))
    if empty_truth:
        print("[주의] 정답 각인이 빈 사진(채점 제외 필요):", empty_truth[:5], "...")
    if missing_img:
        print("[주의] 이미지를 못 찾은 라벨:", missing_img[:5], "...")
    print(f"\n저장: {a.out}")


if __name__ == "__main__":
    main()
