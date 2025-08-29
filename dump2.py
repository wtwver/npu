import sys,os,fcntl,mmap,ctypes,struct,argparse
from gen_parser import Parser,Reg

def _IOC(d,t,n,s):return((d<<30)|(s<<16)|(ord(t)<<8)|n)
def _IOWR(t,n,s):return _IOC(3,t,n,ctypes.sizeof(s))

class drm_version(ctypes.Structure):
    _fields_=[("version_major",ctypes.c_int),("version_minor",ctypes.c_int),("version_patchlevel",ctypes.c_int),("name_len",ctypes.c_size_t),("name",ctypes.POINTER(ctypes.c_char)),("date_len",ctypes.c_size_t),("date",ctypes.POINTER(ctypes.c_char)),("desc_len",ctypes.c_size_t),("desc",ctypes.POINTER(ctypes.c_char))]

class drm_unique(ctypes.Structure):
    _fields_=[("unique_len",ctypes.c_size_t),("unique",ctypes.POINTER(ctypes.c_char))]

class drm_gem_open(ctypes.Structure):
    _fields_=[("name",ctypes.c_uint32),("handle",ctypes.c_uint32),("size",ctypes.c_uint64)]

class rknpu_mem_map(ctypes.Structure):
    _fields_=[("handle",ctypes.c_uint32),("offset",ctypes.c_uint64)]

DRM_IOCTL_VERSION=_IOWR('d',0x00,drm_version)
DRM_IOCTL_GET_UNIQUE=_IOWR('d',0x01,drm_unique)
DRM_IOCTL_GEM_OPEN=_IOWR('d',0x0b,drm_gem_open)
DRM_IOCTL_RKNPU_MEM_MAP=_IOWR('d',0x43,rknpu_mem_map)

def ddf(x,d):
    p=Parser()
    p.parse("",x)
    r={}
    for e in p.file:
        if isinstance(e,Reg):r[e.offset]=e
    with open(d,'rb')as f:
        for i in range(os.path.getsize(d)//8):
            o,v,t=struct.unpack("<hIh",f.read(8))
            if o in r:
                reg=r[o]
                s=f"EMIT(REG_{reg.full_name.upper()}, "
                first=1
                if v==0 or len(reg.bitset.fields)==1:s+=f"0x{v:08x}"
                else:
                    for field in reg.bitset.fields:
                        if field.type=="boolean":
                            if(1<<field.high)&v:
                                if not first:s+=" | "
                                s+=f"{reg.full_name.upper()}_{field.name.upper()}"
                                first=0
                        elif field.type=="uint":
                            mask=(1<<(field.high-field.low+1))-1
                            fv=(v>>field.low)&mask
                            if fv:
                                if not first:s+=" | "
                                s+=f"{reg.full_name.upper()}_{field.name.upper()}({fv})"
                                first=0
                print(s+");")
            else:print(f"{t:x} {o:x} {v:x}")

def dgfl(fd,n):
    try:
        print(f"\n{'='*50}\nProcessing GEM Flink {n}\n{'='*50}")
        g=drm_gem_open()
        g.name=n
        fcntl.ioctl(fd,DRM_IOCTL_GEM_OPEN,g)
        print(f"gem flink {n}: ret=0 handle={g.handle} size={g.size}")
        m=rknpu_mem_map()
        m.handle=g.handle
        fcntl.ioctl(fd,DRM_IOCTL_RKNPU_MEM_MAP,m)
        print(f"memmap returned 0 {hex(m.offset)}")
        instr=mmap.mmap(fd,g.size,mmap.MAP_SHARED,mmap.PROT_READ|mmap.PROT_WRITE,m.offset&0x7fffffff)
        print(f"mmap returned {instr}")
        for i in range(0,g.size,16):
            b=instr[i:i+16]
            h=struct.unpack("<4I",b)
            if all(x==0 for x in h):
                rb=(g.size//16-i//16)
                rb*=16
                print(f"[{i:08x}] = {h[0]:08x} {h[1]:08x} {h[2]:08x} {h[3]:08x}")
                print(f"... {rb} bytes from 0x{i:08x} to 0x{g.size-1:08x} are all zeros")
                break
            print(f"[{i:08x}] = {h[0]:08x} {h[1]:08x} {h[2]:08x} {h[3]:08x}")
        instr.close()
    except OSError as e:print(f"Failed in dump_gem_flink for {n}: {os.strerror(e.errno)}")

def dgfd(fd,n):
    print(f"\n{'='*50}\nProcessing GEM Flink {n} for Register Decode\n{'='*50}")
    try:
        g=drm_gem_open()
        g.name=n
        fcntl.ioctl(fd,DRM_IOCTL_GEM_OPEN,g)
        print(f"gem flink {n}: ret=0 handle={g.handle} size={g.size}")
        m=rknpu_mem_map()
        m.handle=g.handle
        fcntl.ioctl(fd,DRM_IOCTL_RKNPU_MEM_MAP,m)
        print(f"memmap returned 0 {hex(m.offset)}")
        instr=mmap.mmap(fd,g.size,mmap.MAP_SHARED,mmap.PROT_READ|mmap.PROT_WRITE,m.offset&0x7fffffff)
        print(f"mmap returned {instr}")
        os.makedirs("dump",exist_ok=1)
        with open(f"dump/gem{n}-dump","wb")as f:f.write(instr[:])
        with open(f"dump/gem{n}_regdump.bin","wb")as df:
            print(f"Successfully created dump/gem{n}_regdump.bin")
            for i in range(g.size//8):
                instr_val=struct.unpack("<Q",instr[i*8:(i+1)*8])[0]
                if instr_val==0:continue
                val=(instr_val>>16)&0xffffffff
                low=instr_val&0xffff
                dst="noone"
                if(instr_val>>56)&1:dst="PC"
                elif(instr_val>>57)&1:dst="CNA"
                elif(instr_val>>59)&1:dst="CORE"
                elif(instr_val>>60)&1:dst="DPU"
                elif(instr_val>>61)&1:dst="DPU_RDMA"
                elif(instr_val>>62)&1:dst="PPU"
                elif(instr_val>>63)&1:dst="PPU_RDMA"
                if os.path.exists("registers.xml"):
                    try:
                        if not hasattr(dgfd,'parser'):
                            dgfd.parser=Parser()
                            dgfd.parser.parse("","registers.xml")
                        regs={}
                        for e in dgfd.parser.file:
                            if isinstance(e,Reg):regs[e.offset]=e
                        if low in regs:
                            reg=regs[low]
                            reg_info=f"[{8*i+0xffef0000:x}] lsb {instr_val:016x} - {dst}"
                            emit_str=f"EMIT(REG_{reg.full_name.upper()}, "
                            first=1
                            if val==0 or len(reg.bitset.fields)==1:emit_str+=f"0x{val:08x}"
                            else:
                                for field in reg.bitset.fields:
                                    if field.type=="boolean":
                                        if(1<<field.high)&val:
                                            if not first:emit_str+=" | "
                                            emit_str+=f"{reg.full_name.upper()}_{field.name.upper()}"
                                            first=0
                                    elif field.type=="uint":
                                        mask=(1<<(field.high-field.low+1))-1
                                        fv=(val>>field.low)&mask
                                        if fv:
                                            if not first:emit_str+=" | "
                                            emit_str+=f"{reg.full_name.upper()}_{field.name.upper()}({fv})"
                                            first=0
                            emit_str+=");"
                            spacing=" "*(50-len(reg_info))if len(reg_info)<50 else" "
                            print(f"{reg_info}{spacing}{emit_str}")
                        else:print(f"[{8*i+0xffef0000:x}] lsb {instr_val:016x} - {dst} Unknown")
                    except:print(f"[{8*i+0xffef0000:x}] lsb {instr_val:016x} - {dst} Unknown")
                else:print(f"[{8*i+0xffef0000:x}] lsb {instr_val:016x} - {dst} Unknown")
                sl=low if low<=32767 else low-65536
                st=0
                df.write(struct.pack("<h",sl))
                df.write(struct.pack("<I",val))
                df.write(struct.pack("<h",st))
        print(f"Dumped {g.size//8} register commands to dump/gem{n}_regdump.bin")
        instr.close()
    except OSError as e:
        print(f"Failed in dump_gem_for_decode for {n}: {os.strerror(e.errno)}")

if __name__=="__main__":
    p=argparse.ArgumentParser(description='NPU Register Dumper and Decoder')
    p.add_argument('--xml',type=str,help='XML register definition file')
    p.add_argument('--dump',type=str,help='Binary dump file to decode')
    p.add_argument('gems',nargs='*',type=int,help='GEM object numbers to dump (default: 1, 2)')
    args=p.parse_args()
    if args.xml and args.dump:
        if not os.path.exists(args.xml):print(f"Error: XML file '{args.xml}' not found",file=sys.stderr);sys.exit(1)
        if not os.path.exists(args.dump):print(f"Error: Dump file '{args.dump}' not found",file=sys.stderr);sys.exit(1)
        ddf(args.xml,args.dump)
        sys.exit(0)
    try:
        fd=os.open("/dev/dri/card1",os.O_RDWR)
    except OSError as e:
        print(f"Failed to open /dev/dri/card1: {os.strerror(e.errno)}")
        sys.exit(1)
    try:
        name_buf=(ctypes.c_char*256)()
        date_buf=(ctypes.c_char*256)()
        desc_buf=(ctypes.c_char*256)()
        dv=drm_version()
        dv.version_major=0
        dv.version_minor=0
        dv.version_patchlevel=0
        dv.name_len=256
        dv.name=ctypes.cast(name_buf,ctypes.POINTER(ctypes.c_char))
        dv.date_len=256
        dv.date=ctypes.cast(date_buf,ctypes.POINTER(ctypes.c_char))
        dv.desc_len=256
        dv.desc=ctypes.cast(desc_buf,ctypes.POINTER(ctypes.c_char))
        fcntl.ioctl(fd,DRM_IOCTL_VERSION,dv)
        name_str=ctypes.string_at(dv.name,dv.name_len).decode('utf-8',errors='ignore').rstrip('\x00')
        date_str=ctypes.string_at(dv.date,dv.date_len).decode('utf-8',errors='ignore').rstrip('\x00')
        desc_str=ctypes.string_at(dv.desc,dv.desc_len).decode('utf-8',errors='ignore').rstrip('\x00')
        print(f"drm name is {name_str} - {date_str} - {desc_str}")
        unique_buf=(ctypes.c_char*256)()
        du=drm_unique()
        du.unique_len=256
        du.unique=ctypes.cast(unique_buf,ctypes.POINTER(ctypes.c_char))
        fcntl.ioctl(fd,DRM_IOCTL_GET_UNIQUE,du)
        unique_str=ctypes.string_at(du.unique,du.unique_len).decode('utf-8',errors='ignore').rstrip('\x00')
        print(f"du is {unique_str}")
    except OSError as e:
        print(f"Error in DRM ioctl: {os.strerror(e.errno)}")
        os.close(fd)
        sys.exit(2)
    gems=args.gems if args.gems else[1,2]
    print("Dumping specified GEM objects...")
    for n in gems:
        if n>0:
            print(f"\n=== Processing GEM {n} ===")
            dgfl(fd,n)
            dgfd(fd,n)
        else:print(f"Invalid GEM number: {n}",file=sys.stderr)
    os.close(fd)
