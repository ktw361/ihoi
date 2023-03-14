import argparse
import numpy as np
from libzhifan import io

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--set', type=str, choice=['all', 'rnd100', 'eval100'], required=True)
    args = parser.parse_args()
    return args


def report_for_set(set_name, df):
    infiles = dict(
        eval100 = '/home/skynet/Zhifan/epic_analysis/hos/tools/eval_100_Feb25.json',
        all='/home/skynet/Zhifan/epic_analysis/hos/tools/model-input-Feb03.json',
        rnd100='/home/skynet/Zhifan/epic_analysis/hos/tools/eval_rand100_Mar05.json',
    )
    infile = infiles[set_name]
    infos = io.read_json(infile)
    vid2cat = dict()
    for info in infos:
        vid = '%s_%s_%s' % (info['vid'], info['start'], info['end'])
        vid2cat[vid] = info['cat']

    missing = set(vid2cat.keys()) - set(df['vid_key'])
    print(f"Missing {len(missing)} videos") # : {missing}")

    oiou = np.mean(df['oious']) * 100
    hiou = np.mean(df['hious']) * 100
    pd = np.mean(df['pds'])
    iv = np.mean(df['ivs'])
    cats = ['plate', 'bowl', 'bottle', 'cup', 'mug', 'can']
    cat_ious = dict()
    df['cat'] = df['vid_key'].apply(lambda x: vid2cat[x])
    for cat in cats:
        cat_df = df[df['cat'] == cat]
        oiou = np.mean(cat_df['oious']) * 100
        cat_ious[cat] = oiou
    
    print("hiou, oiou, plate, bowl, bottle, cup, mug, can, iv, pd")
    print(f"{hiou:.1f} & {oiou:.1f} & {cat_ious['plate']:.1f} & {cat_ious['bowl']:.1f} & {cat_ious['bottle']:.1f} & {cat_ious['cup']:.1f} & {cat_ious['mug']:.1f} & {cat_ious['can']:.1f} & {iv:.1f} & {pd:.1f}")


def main(args):
    df = io.read_csv(args.csv)
    report_for_set(args.set, df)


if __name__ == '__main__':
    main(parse_args())