"""Set the GitHub repository description, website, and topics after gh auth login."""
import os
import subprocess

REPOSITORY = 'alalbiol/sentinel-landcover-inference'
DESCRIPTION = ('BigEarthNet 19-class land-cover classification of Sentinel-1/2 patches with three '
               'TerraFM-B models: inference, evaluation and training code; weights on EOTDL.')
HOMEPAGE = 'https://www.eotdl.com/models/sentinel-landcover-inference'
TOPICS = ['land-cover', 'bigearthnet', 'sentinel-2', 'sentinel-1', 'foundation-models',
          'terrafm', 'multi-label-classification', 'pytorch', 'remote-sensing']


def main():
    env = {k: v for k, v in os.environ.items() if k not in ('GH_TOKEN', 'GITHUB_TOKEN', 'GH_DEBUG')}
    subprocess.run(['gh', 'auth', 'status', '--hostname', 'github.com'], env=env, check=True)
    command = ['gh', 'repo', 'edit', REPOSITORY, '--description', DESCRIPTION, '--homepage', HOMEPAGE]
    for topic in TOPICS:
        command.extend(['--add-topic', topic])
    subprocess.run(command, env=env, check=True)
    print(f'Updated About fields for {REPOSITORY}.')


if __name__ == '__main__':
    main()
