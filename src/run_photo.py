r"""사진 1장 → 알약마다 약 이름 후보. 가상환경이 둘이라 세 번에 나눠 실행한다 (저장소 맨 위 D:\dl_deep_imgps 에서).

    .\.venv\Scripts\python.exe            src\run_photo.py --step crop  --photo exp_cns\data_user\photo1.png   # 0단계
    .\.venv_paddle_gpu\Scripts\python.exe src\run_photo.py --step boxes --photo exp_cns\data_user\photo1.png   # 1~2단계
    .\.venv\Scripts\python.exe            src\run_photo.py --step read  --photo exp_cns\data_user\photo1.png   # 3~6단계

저장: exp_cns\results\app_v1\<사진 이름>\  crops\pill_N.png, crops.csv, check_crop.png (crop) / boxes.json (boxes) /
      result.csv, result.png (read). 이미 있는 파일은 덮어쓰지 않고 멈춘다. 사진이 든 PNG 는 .gitignore 로 올라가지 않는다.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pill_id import config as C                                        # noqa: E402
from pill_id.imageio import fit_side, imread, imwrite, rotate_color     # noqa: E402


NO_PILL = "알약을 찾지 못했습니다. 배경이 단순한 곳에서 알약이 사진 가장자리에 닿지 않게 다시 촬영해 주세요."


def folder(a):
    return Path(a.out) / Path(a.photo).stem


def save_csv(path, rows, fields):
    """임시 파일에 다 쓴 뒤 이름을 바꿔 저장 → 도중에 멈추면 결과 파일이 생기지 않아 다시 실행할 수 있음."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)
    tmp.replace(path)


def save_json(path, obj):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)
    tmp.replace(path)


def load_crops(out):
    rows = list(csv.DictReader(open(out / "crops.csv", encoding="utf-8-sig")))
    for r in rows:
        r["img"] = imread(str(out / "crops" / r["file_name"]))
    return rows


# ---------------------------------------------------------------------
# 0단계 (.venv)
# ---------------------------------------------------------------------
def step_crop(a):
    from pill_id import s0_crop
    out = folder(a)
    if (out / "crops.csv").exists():
        raise SystemExit(f"이미 있음: {out} (덮어쓰지 않음)")
    bgr = imread(a.photo)
    pills, drop = s0_crop.crop_pills(bgr)
    rows = []
    for p in pills:
        name = f"pill_{p['pill']}.png"
        imwrite(str(out / "crops" / name), p["img"])
        x0, y0, x1, y1 = p["box"]
        rows.append(dict(pill=p["pill"], file_name=name, x0=x0, y0=y0, x1=x1, y1=y1, w=x1 - x0, h=y1 - y0))
    vis = s0_crop.draw_check(bgr, pills, drop)
    imwrite(str(out / "check_crop.png"), fit_side(vis, 1600) if max(vis.shape[:2]) > 1600 else vis)
    save_csv(out / "crops.csv", rows, ["pill", "file_name", "x0", "y0", "x1", "y1", "w", "h"])   # 마지막에 저장 = 완료 표시
    print(f"알약 {len(pills)}개, 버린 덩어리 {len(drop)}개 ({', '.join(r for _, r in drop) or '-'}) → {out}")
    if not pills:
        print(NO_PILL + f" (확인 그림: {out / 'check_crop.png'})")


# ---------------------------------------------------------------------
# 1~2단계 (.venv_paddle_gpu) — torch 를 import 하지 않음
# ---------------------------------------------------------------------
def step_boxes(a):
    from pill_id import s1_orient, s2_detect
    out = folder(a)
    if (out / "boxes.json").exists():
        raise SystemExit(f"이미 있음: {out / 'boxes.json'}")
    crops = load_crops(out)
    if not crops:                                     # 모델을 불러오기 전에 멈춤
        raise SystemExit(NO_PILL)
    det = s2_detect.Detector()
    res = s2_detect.detect_all(det, [(r["file_name"], r["img"]) for r in crops], s1_orient.angles)
    save_json(out / "boxes.json", res)
    for name, v in res.items():
        print(f"  {name}: {len(v)}방향, 상자 수 {[len(x['boxes']) for x in v.values()]}")
    print(f"저장: {out / 'boxes.json'} (Paddle 장치 {det.device})")


# ---------------------------------------------------------------------
# 3~6단계 (.venv)
# ---------------------------------------------------------------------
def step_read(a):
    import cv2
    from pill_id import s1_orient, s2_detect, s3_read, s4_appearance, s5_match, s6_display
    out = folder(a)
    if (out / "result.csv").exists():
        raise SystemExit(f"이미 있음: {out / 'result.csv'}")
    crops = load_crops(out)
    if not crops:                                     # 모델을 불러오기 전에 멈춤 (빈 목록이면 CNN 입력 np.stack 이 실패)
        raise SystemExit(NO_PILL)
    boxes = s2_detect.load_boxes(out / "boxes.json")
    reader = s3_read.Reader(str(C.parseq_weights()))
    probs = s4_appearance.AppearanceCNN().predict([r["img"] for r in crops])
    db = s5_match.load_db()
    byid = {d["drug_N"]: d for d in db}

    def lab(d):
        return f"{byid[d]['name']} [{'/'.join(byid[d]['imprints']) or '-'}]"

    rows, panels = [], []
    for r, prob in zip(crops, probs):
        angs, _ = s1_orient.angles(r["img"])
        dirs, best, img = s3_read.read_pill(r["img"], angs, lambda ang, fn=r["file_name"]: boxes(fn, ang), reader)
        pred, ranked = s5_match.match(dirs, best, prob, db)
        kind, shown = s6_display.shown(pred, ranked)
        sh, fo, co = s4_appearance.summary(prob)
        more = f" ; … 외 {len(shown) - 10}개" if len(shown) > 10 else ""
        row = dict(pill=r["pill"], aligned=int(len(angs) == 2), n_dirs=len(angs), read=best["text"], angle=best["ang"],
                   conf=round(best["conf"], 1), n_boxes=0 if best["fallback"] else len(best["boxes"]),
                   cnn_shape=sh, cnn_form=fo, cnn_colors=co,
                   answer=lab(pred) if pred else "판정 없음", flow=s6_display.flow(kind, shown),
                   shown=" ; ".join(lab(d) for d in shown[:10]) + more,
                   top5=" ; ".join(f"{lab(d)} {v:.3f}" for d, v in ranked[:5]),
                   all_dirs=" | ".join(f"{d['ang']}:{d['text']}:{d['conf']:.1f}" for d in dirs))
        rows.append(row)
        rot = rotate_color(img, best["ang"]).copy()
        for (x0, y0, x1, y1), t in zip(best["boxes"], best["texts"]):
            cv2.rectangle(rot, (x0, y0), (x1, y1), (0, 0, 255) if best["fallback"] else (0, 200, 0), 2)
            cv2.putText(rot, t or "-", (x0, max(14, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        panels.append((r, rot, row, ranked))
    # 그림을 먼저 저장하고 CSV 를 마지막에 → 그림 저장이 실패하면 result.csv 가 없어 그대로 다시 실행할 수 있음
    draw_result(out, a.photo, panels, lab)
    save_csv(out / "result.csv", rows, list(rows[0]))
    for row in rows:
        print(f"\n#{row['pill']}  읽음 '{row['read']}' (방향 {row['angle']}°, 확신도 {row['conf']}, 상자 {row['n_boxes']})"
              f"  CNN {row['cnn_shape']}/{row['cnn_form']}/{row['cnn_colors'] or '-'}")
        print(f"   답: {row['answer']}  /  표시: {row['flow']}")
        print(f"   1~5등: {row['top5']}")


def draw_result(out, photo, panels, lab):
    """결과 그림: 맨 위 원본 + 찾은 알약, 아래 알약마다 자른 사진 | 고른 방향 + 상자 | 결과 글."""
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Malgun Gothic"
    n = len(panels)
    fig = plt.figure(figsize=(16, 4.5 + 4.2 * n))
    gs = fig.add_gridspec(1 + n, 3, height_ratios=[4.5] + [4.2] * n, width_ratios=[1, 1, 2.2])
    ax = fig.add_subplot(gs[0, :])
    ax.imshow(cv2.cvtColor(imread(str(out / "check_crop.png")), cv2.COLOR_BGR2RGB))
    ax.set_title(f"{Path(photo).name}: 찾은 알약 {n}개 (파랑 네모 = 자른 영역, 주황 = 버린 덩어리)", fontsize=13)
    ax.axis("off")
    for k, (r, rot, row, ranked) in enumerate(panels, 1):
        a1, a2, a3 = (fig.add_subplot(gs[k, j]) for j in range(3))
        a1.imshow(cv2.cvtColor(r["img"], cv2.COLOR_BGR2RGB))
        a1.set_title(f"#{row['pill']} 자른 알약", fontsize=11)
        a2.imshow(cv2.cvtColor(rot, cv2.COLOR_BGR2RGB))
        a2.set_title(f"고른 방향 {row['angle']}° · 상자 {row['n_boxes']}개", fontsize=11)
        lines = [f"읽은 각인: {row['read'] or '(없음)'}   (확신도 {row['conf']}, {'긴 알약 축 2방향' if row['aligned'] else '12방향'})",
                 f"겉모습 CNN: {row['cnn_shape']} / {row['cnn_form']} / {row['cnn_colors'] or '-'}",
                 f"답: {row['answer']}", f"표시: {row['flow']}", ""]
        lines += [f"  {'★' if j == 0 else '·'} {lab(d)} {v:.3f}" for j, (d, v) in enumerate(ranked[:5])]
        a3.text(0, 1, "\n".join(lines), va="top", fontsize=11, family="Malgun Gothic")
        for x in (a1, a2, a3):
            x.axis("off")
    fig.tight_layout()
    fig.savefig(out / "result.png", dpi=90)
    print(f"그림: {out / 'result.png'}")


def main():
    ap = argparse.ArgumentParser(description="사진 1장 → 알약마다 약 이름 후보 (단계: crop → boxes → read)")
    ap.add_argument("--step", required=True, choices=["crop", "boxes", "read"])
    ap.add_argument("--photo", required=True)
    ap.add_argument("--out", default=str(C.OUT_ROOT), help="결과 상위 폴더")
    a = ap.parse_args()
    {"crop": step_crop, "boxes": step_boxes, "read": step_read}[a.step](a)


if __name__ == "__main__":
    main()
