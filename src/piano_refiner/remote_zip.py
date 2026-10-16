"""Read selected official ZIP entries over validated HTTP byte ranges."""
import io,re,urllib.request,time

class HTTPRangeReader(io.RawIOBase):
    def __init__(self,url,budget=4*1024**3):
        self.url=url;self.pos=0;self.used=0;self.budget=budget
        with urllib.request.urlopen(urllib.request.Request(url,headers={'Range':'bytes=0-0','Accept-Encoding':'identity'}),timeout=60) as r:
            match=re.fullmatch(r'bytes 0-0/(\d+)',r.headers.get('Content-Range',''))
            if r.status!=206 or not match:raise IOError('Server does not support byte ranges; refusing full archive download')
            self.size=int(match.group(1));r.read(1)
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        p=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        if p<0:raise ValueError('Negative seek')
        self.pos=p;return p
    def read(self,n=-1):
        n=max(0,self.size-self.pos) if n<0 else min(n,max(0,self.size-self.pos))
        if not n:return b''
        if n>256*1024**2 or self.used+n>self.budget:raise IOError('Download safety budget exceeded')
        start=self.pos;end=start+n-1
        for attempt in range(3):
            try:
                req=urllib.request.Request(self.url,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'})
                with urllib.request.urlopen(req,timeout=90) as r:
                    if r.status!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{self.size}':raise IOError('Range response mismatch')
                    data=r.read(n+1)
                if len(data)!=n:raise IOError('Truncated range')
                self.used+=n;self.pos+=n;return data
            except Exception:
                if attempt==2:raise
                time.sleep(2*(attempt+1))
