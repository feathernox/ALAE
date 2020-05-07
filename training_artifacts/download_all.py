import argparse
import dlutils
import os
from packaging import version


if __name__ == "__main__":
    if not hasattr(dlutils, "__version__") or version.parse(dlutils.__version__) < version.parse("0.0.11"):
        raise RuntimeError('Please update dlutils: pip install dlutils --upgrade')

    parser = argparse.ArgumentParser(description='Download all ALAE checkpoints.')
    parser.add_argument('--path', type=str, default='training_artifacts', help='path to save checkpoints')
    args = parser.parse_args()

    ffhq_dir = os.path.join(args.path, 'ffhq')

    try:
        dlutils.download.from_google_drive('170Qldnn28IwnVm9CQEq1AZhVsK7PJ0Xz', directory=ffhq_dir)
        dlutils.download.from_google_drive('1QESywJW8N-g3n0Csy0clztuJV99g8pRm', directory=ffhq_dir)
        dlutils.download.from_google_drive('18BzFYKS3icFd1DQKKTeje7CKbEKXPVug', directory=ffhq_dir)
    except IOError:
        dlutils.download.from_url('https://alaeweights.s3.us-east-2.amazonaws.com/ffhq/model_submitted.pth',
                                  directory=ffhq_dir)
        dlutils.download.from_url('https://alaeweights.s3.us-east-2.amazonaws.com/ffhq/model_194.pth',
                                  directory=ffhq_dir)
        dlutils.download.from_url('https://alaeweights.s3.us-east-2.amazonaws.com/ffhq/model_157.pth',
                                  directory=ffhq_dir)

    celeba_dir = os.path.join(args.path, 'celeba')
    try:
        dlutils.download.from_google_drive('1T4gkE7-COHpX38qPwjMYO-xU-SrY_aT4',
                                           directory=celeba_dir)
    except IOError:
        dlutils.download.from_url('https://alaeweights.s3.us-east-2.amazonaws.com/celeba/model_final.pth',
                                  directory=celeba_dir)

    bedroom_dir = os.path.join(args.path, 'bedroom')
    try:
        dlutils.download.from_google_drive('1gmYbc6Z8qJHJwICYDsB4aBMxXjnKeXA_', directory=bedroom_dir)
    except IOError:
        dlutils.download.from_url('https://alaeweights.s3.us-east-2.amazonaws.com/bedroom/model_final.pth',
                                  directory=bedroom_dir)

    celeba_hq256_dir = os.path.join(args.path, 'celeba-hq256')
    try:
        dlutils.download.from_google_drive('1ihJvp8iJWcLxTIjkV5cyA7l9TrxlUPkG', directory=celeba_hq256_dir)
        dlutils.download.from_google_drive('1gFQsGCNKo-frzKmA3aCvx07ShRymRIKZ', directory=celeba_hq256_dir)
    except IOError:
        dlutils.download.from_url('https://alaeweights.s3.us-east-2.amazonaws.com/celeba-hq256/model_262r.pth',
                                  directory=celeba_hq256_dir)
        dlutils.download.from_url('https://alaeweights.s3.us-east-2.amazonaws.com/celeba-hq256/model_580r.pth',
                                  directory=celeba_hq256_dir)
