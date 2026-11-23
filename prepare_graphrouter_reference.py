#!/usr/bin/env python3
"""Acquire the authors' released artifacts and validate the reference dataset."""
import argparse
import hashlib
from html.parser import HTMLParser
import http.cookiejar
import json
from pathlib import Path
import shutil
import urllib.parse
import urllib.request

import torch
from router.graphrouter_reference import BLOBS, DRIVE_ID, UPSTREAM, prepare, sha256, verify_blob


class DownloadForm(HTMLParser):
    def __init__(self):
        super().__init__(); self.action=None; self.fields={}; self.in_form=False
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag=='form' and attrs.get('id')=='download-form':
            self.action=attrs.get('action'); self.in_form=True
        if tag=='input' and self.in_form and attrs.get('name'):
            self.fields[attrs['name']]=attrs.get('value','')
    def handle_endtag(self,tag):
        if tag=='form': self.in_form=False


def download(url, target, drive=False):
    target=Path(target); partial=target.with_name(target.name+'.part')
    opener=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    for attempt in range(3):
        request=urllib.request.Request(url,headers={'User-Agent':'GraphRouter-reference-artifact-reader/1.0'})
        with opener.open(request,timeout=60) as response:
            if 'text/html' in response.headers.get('Content-Type',''):
                html=response.read().decode('utf-8',errors='replace')
                form=DownloadForm(); form.feed(html)
                if not drive or not form.action:
                    raise RuntimeError('artifact download returned HTML (permission/quota/sign-in page); no fallback or LLM regeneration')
                parsed=urllib.parse.urlparse(form.action)
                if parsed.scheme!='https' or parsed.hostname not in ('drive.usercontent.google.com','drive.google.com','docs.google.com'):
                    raise RuntimeError('unexpected download confirmation host')
                url=form.action+('?' if not parsed.query else '&')+urllib.parse.urlencode(form.fields)
                continue
            with partial.open('wb') as f:
                shutil.copyfileobj(response,f)
        partial.replace(target)
        return
    raise RuntimeError('could not complete public Drive download confirmation')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--artifact-dir',type=Path,default=Path('data/graphrouter_reference'))
    ap.add_argument('--router-data-path',type=Path,help='Existing copy of the authors\' CSV; not ProMoRouter logs')
    args=ap.parse_args(); root=args.artifact_dir;root.mkdir(parents=True,exist_ok=True)
    records={}
    for name in BLOBS:
        url=f'https://raw.githubusercontent.com/ulab-uiuc/GraphRouter/{UPSTREAM}/configs/{name}'
        path=root/name
        if not path.exists():
            print(f'Downloading pinned upstream {name}',flush=True);download(url,path)
        verify_blob(path)
        records[name]=dict(url=url,sha256=sha256(path))
    csv_path=root/'router_data.csv'
    if args.router_data_path:
        if csv_path.exists() and sha256(csv_path)!=sha256(args.router_data_path):
            raise FileExistsError('existing reference CSV differs from supplied file')
        if not csv_path.exists():shutil.copyfile(args.router_data_path,csv_path)
    elif not csv_path.exists():
        print('Downloading authors\' released router_data.csv (no LLM calls)',flush=True)
        download(f'https://drive.usercontent.google.com/download?id={DRIVE_ID}&export=download&confirm=t',csv_path,drive=True)
    records['router_data.csv']=dict(
        release_link=f'https://drive.google.com/file/d/{DRIVE_ID}/view',
        source='supplied_existing_copy' if args.router_data_path else 'authors_release_link_or_cached_copy',
        sha256=sha256(csv_path))
    (root/'artifact_sources.json').write_text(json.dumps(records,indent=2)+'\n')
    data=prepare(root)
    output=root/'prepared.pt'
    if output.exists():
        previous=torch.load(output,map_location='cpu',weights_only=True)
        if (previous['metadata']!=data['metadata'] or previous['splits']!=data['splits']
                or previous['task_names']!=data['task_names']
                or any(not torch.equal(previous[k],data[k]) for k in ('query','task','model','performance','cost'))):
            raise FileExistsError('existing prepared bundle differs; preserve it before proceeding')
        print(f'Validated existing prepared bundle: {output}')
    else:
        torch.save(data,output);print(f'Prepared reference: {output}')
    print(json.dumps(data['metadata'],indent=2))


if __name__=='__main__':main()
