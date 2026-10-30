"""Side-by-side video of the real sensor against each proxy variant.

The real panel is NOT frame-matched to the proxy panels - it is a different flight in a
different arena, shown at the same colour scale so the texture, density and polarity
structure can be compared by eye. The numbers under each panel are that panel's own
per-frame statistics.

  python3 bench/make_compare_video.py --tags baseline step1_bg
"""
import argparse, glob, os, subprocess, sys
import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'analysis'))
import ppofly

FPS, HOLD, S = 30, 2, 360
INK, MUTED, BLUE, RED = (26, 26, 30), (122, 124, 134), (26, 102, 210), (198, 48, 44)


def font(sz, bold=False):
    for n in (f'dejavu/DejaVuSans{"-Bold" if bold else ""}.ttf',):
        p = f'/usr/share/fonts/truetype/{n}'
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def ev_rgb(frame_u8, size=S, vmax=12.0):
    d = np.asarray(frame_u8, np.float32).squeeze() - 128.0
    rgb = (colormaps['bwr_r'](np.clip(d / vmax, -1, 1) * .5 + .5)[..., :3] * 255).astype(np.uint8)
    return np.asarray(Image.fromarray(rgb).resize((size, size), Image.NEAREST))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tags', nargs='+', default=['baseline', 'step1_bg'])
    ap.add_argument('--out', default=os.path.join(HERE, 'out', 'compare.mp4'))
    A = ap.parse_args()
    os.makedirs(os.path.dirname(A.out), exist_ok=True)

    real = ppofly.read(sorted(glob.glob(os.path.join(
        ROOT, 'analysis/data/onboard_events/**/*.bin'), recursive=True))[7])['frames']
    panels = [('real sensor', real, (28, 130, 80))]
    for t in A.tags:
        f = os.path.join(HERE, 'runs', t + '.npz')
        if os.path.exists(f):
            arr = np.load(f)['frames']
            if arr.ndim == 4:                      # [N,H,W,1] -> [N,H,W]
                arr = arr[..., 0]
            assert arr.ndim == 3 and arr.shape[1] == arr.shape[2], f'{t}: bad shape {arr.shape}'
            panels.append((t, arr, BLUE))
    n = max(len(p[1]) for p in panels)
    W = 90 + len(panels) * (S + 60)
    H = S + 260
    frames = []
    for i in range(n):
        img = Image.new('RGB', (W, H), (255, 255, 255))
        d = ImageDraw.Draw(img)
        d.text((W // 2, 34), 'Real sensor vs proxy variants', font=font(30, True),
               fill=INK, anchor='ma')
        d.text((W // 2, 76), 'colour scale +/-12 counts   |   blue = ON, red = OFF   |   '
               'the real panel is a different flight, not frame-matched',
               font=font(15), fill=MUTED, anchor='ma')
        for j, (name, arr, col) in enumerate(panels):
            x = 60 + j * (S + 60)
            fr = arr[i % len(arr)]
            img.paste(Image.fromarray(ev_rgb(fr)), (x, 130))
            d.rectangle([x - 1, 129, x + S, 130 + S], outline=(214, 214, 222))
            d.text((x + S // 2, 108), name, font=font(20, True), fill=col, anchor='ma')
            dev = fr.astype(np.int16) - 128
            act = np.abs(dev) > .5
            on = (dev > .5).sum() / max(act.sum(), 1) * 100
            d.text((x + S // 2, S + 150), f'{act.mean()*100:.0f} % active    {on:.0f} % ON',
                   font=font(19), fill=INK, anchor='ma')
        d.text((W // 2, H - 48), f'frame {i+1}/{n}   ({(i+1)/13.3:.1f} s at 13.3 Hz)',
               font=font(16), fill=MUTED, anchor='ma')
        frames.extend([np.asarray(img)] * HOLD)

    h, w = frames[0].shape[:2]
    w -= w % 2; h -= h % 2
    p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
                          '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', str(FPS), '-i', '-',
                          '-c:v', 'libx264', '-preset', 'slow', '-crf', '20',
                          '-pix_fmt', 'yuv420p', '-movflags', '+faststart', A.out],
                         stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f[:h, :w].astype(np.uint8).tobytes())
    p.stdin.close()
    assert p.wait() == 0
    print(f'{A.out}  ({len(frames)} frames, {len(frames)/FPS:.1f} s, '
          f'{os.path.getsize(A.out)/1e6:.1f} MB)')


if __name__ == '__main__':
    main()
