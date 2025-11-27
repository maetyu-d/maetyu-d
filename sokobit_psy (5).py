
import math
import random
from dataclasses import dataclass, field

import pygame

try:
    import sounddevice as sd
    HAS_SD = True
except Exception:
    sd = None
    HAS_SD = False

def clamp(v, lo=0, hi=255):
    return max(lo, min(hi, v))

def blend(c1, c2, a):
    return tuple(int((1 - a) * c1[i] + a * c2[i]) for i in range(3))

def cycle_color(rgb, t, strength=0.4):
    r, g, b = rgb
    r = clamp(int(r + strength * 80 * math.sin(t * 0.9 + 0.0)))
    g = clamp(int(g + strength * 80 * math.sin(t * 0.9 + 2.1)))
    b = clamp(int(b + strength * 80 * math.sin(t * 0.9 + 4.2)))
    return (r, g, b)

OPCODES = {
    "NOP": 0, "XOR": 1, "AND": 2, "OR": 3, "NOT": 4,
    "SHL": 5, "SHR": 6, "ADD": 7, "SUB": 8, "ROL": 9,
    "ROR": 10, "MOV": 11, "LOADC": 12, "NOISE": 13,
    "STEP": 14, "MUL": 15,
}

OPCODE_COLORS = {
    0:  (30, 30, 40),
    1:  (255, 40, 255),
    2:  (60, 255, 150),
    3:  (60, 220, 255),
    4:  (250, 250, 250),
    5:  (255, 180, 60),
    6:  (255, 230, 80),
    7:  (100, 140, 255),
    8:  (60, 170, 230),
    9:  (200, 80, 255),
    10: (255, 120, 220),
    11: (130, 230, 255),
    12: (255, 210, 140),
    13: (255, 70, 70),
    14: (170, 255, 90),
    15: (255, 90, 140),
}

def rol16(x, n=1):
    x &= 0xFFFF
    n &= 15
    return ((x << n) | (x >> (16 - n))) & 0xFFFF

def ror16(x, n=1):
    x &= 0xFFFF
    n &= 15
    return ((x >> n) | (x << (16 - n))) & 0xFFFF

@dataclass
class BitGridProgram:
    grid: list = field(default_factory=lambda: [[0] * 16 for _ in range(16)])
    @classmethod
    def empty(cls):
        return cls()

PENT = [0, 3, 5, 7, 10]
ROOT_MIDI = 52

def midi_to_inc(midi, sr):
    freq = 440.0 * (2.0 ** ((midi - 69) / 12.0))
    return freq / sr

@dataclass
class BitGridVM:
    program: BitGridProgram
    hardware_sample_rate: int = 44100

    def __post_init__(self):
        self.registers = [0]*16
        self.registers[1] = 180
        self.registers[2] = 230
        self.registers[10] = 90
        self.registers[3] = 0xAAAA
        self.registers[4] = 0x0F0F
        self.registers[5] = 0x3333
        self.registers[6] = 32
        self.registers[7] = 128
        self.registers[11] = 0x4000
        self.registers[12] = 0x8000
        self.registers[14] = 0x8000
        self.registers[0]=0

        self.lfsr=0xACE1
        self.seq_interval=256
        self.mod_interval=64
        self.chaos_interval=512
        self._seq=0
        self._mod=0
        self._chaos=0
        self.phase_scale=1.0
        self._filter_state=0.0
        self._filter_alpha=0.2
        self._am_depth=0.3
        self.morph=0.5
        self.n1=self.n2=self.n3=0.0
        self.b1=self.b2=self.b3=0.0

    def _noise16(self):
        bit=((self.lfsr>>0)^(self.lfsr>>2)^(self.lfsr>>3)^(self.lfsr>>5))&1
        self.lfsr=(self.lfsr>>1)|(bit<<15)
        return self.lfsr&0xFFFF

    def run_row(self,r):
        g=self.program.grid
        regs=self.registers
        acc=regs[r]&0xFFFF
        for c in range(16):
            op=g[r][c]
            src=regs[c]&0xFFFF
            if op==0: continue
            elif op==1: acc^=src
            elif op==2: acc&=src
            elif op==3: acc|=src
            elif op==4: acc=(~acc)&0xFFFF
            elif op==5: acc=(acc<<1)&0xFFFF
            elif op==6: acc=(acc>>1)&0xFFFF
            elif op==7: acc=(acc+src)&0xFFFF
            elif op==8: acc=(acc-src)&0xFFFF
            elif op==9: acc=rol16(acc,src&3)
            elif op==10: acc=ror16(acc,src&3)
            elif op==11: acc=src
            elif op==12: acc=((r<<4)|c)&0xFFFF
            elif op==13: acc^=self._noise16()
            elif op==14: regs[0]=(regs[0]+(src&0xF)+1)&0xF
            elif op==15: acc=((acc*(src&0xFF))>>4)&0xFFFF
        regs[r]=acc

    def update_meta(self):
        r=self.registers
        raw=r[6]&0xFF
        self.seq_interval=max(4,min(8192,4+raw*32))
        rawm=r[7]&0xFF
        self.mod_interval=max(1,min(2048,1+rawm*8))
        rawc=r[11]&0xFF
        self.chaos_interval=max(8,min(4096,8+rawc*16))
        self.phase_scale=0.2+(rawm/255.0)*3.0
        rawf=r[12]&0xFF
        self._filter_alpha=0.02+(rawf/255.0)*0.45
        self._am_depth=(rawc/255.0)*0.6
        self.morph=(r[14]&0xFF)/255.0

    def nasty(self):
        r=self.registers
        inc1=(r[1]&0xFF)*self.phase_scale/65536.0
        inc2=(r[2]&0xFF)*self.phase_scale*0.75/65536.0
        inc3=(r[10]&0xFF)*self.phase_scale*0.5/65536.0
        self.n1=(self.n1+inc1)%1.0
        self.n2=(self.n2+inc2)%1.0
        self.n3=(self.n3+inc3)%1.0
        pha=int(self.n1*256)&0xFF
        phb=int(self.n2*256)&0xFF
        phc=int(self.n3*256)&0xFF
        w1=pha^((r[3]>>8)&0xFF)
        w2=phb&((r[4]>>8)&0xFF)
        tri=phc if phc<128 else 255-phc
        w3=(tri+((r[5]>>8)&0x3F))&0xFF
        mix=(w1+w2+w3)&0xFF
        if (r[3]&0x8000) and (self.lfsr&0xF)==0:
            mix^=(self._noise16()>>8)&0xFF
        amsrc=(r[2]>>8)&0xFF
        am=1.0+self._am_depth*((amsrc/127.5)-1.0)
        val=(mix-128)/128.0
        val*=am
        fm=((r[10]>>8)&0xFF)-128
        val+=fm/1024.0
        return math.tanh(val*1.2)

    def beaut(self):
        r=self.registers
        sr=self.hardware_sample_rate
        def q(val):
            deg=val&0xF
            st=PENT[deg%len(PENT)]
            octv=(val>>4)&0x3
            return ROOT_MIDI+st+12*octv

        m1=q(r[1])
        m2=q(r[2])
        m3=q(r[10])-12
        inc1=midi_to_inc(m1,sr)*self.phase_scale
        inc2=midi_to_inc(m2,sr)*self.phase_scale*0.99
        inc3=midi_to_inc(m3,sr)*self.phase_scale*0.5
        self.b1=(self.b1+inc1)%1.0
        self.b2=(self.b2+inc2)%1.0
        self.b3=(self.b3+inc3)%1.0
        ph1=self.b1
        ph2=self.b2
        ph3=self.b3
        s1=math.sin(2*math.pi*ph1)
        s2=math.sin(2*math.pi*(ph2+0.01))
        tri=2*abs(2*(ph3-math.floor(ph3+0.5)))-1.0
        fm_amt=((r[3]>>8)&0x3F)/4096.0
        s1f=math.sin(2*math.pi*(ph1+fm_amt*tri))
        mix=0.55*s1f+0.3*s2+0.25*tri
        amsrc=((r[2]>>8)&0xFF)/255.0
        mix*=1.0+self._am_depth*0.5*(amsrc-0.5)*2.0
        noise=((self._noise16()&0xFF)/127.5-1.0)*0.02
        mix+=noise
        return math.tanh(mix*1.1)

    def tick(self):
        r=self.registers
        if self._seq<=0:
            self.run_row(r[0]&0xF)
            self._seq=self.seq_interval
        self._seq-=1
        if self._mod<=0:
            self.run_row((r[0]+5)&0xF)
            self._mod=self.mod_interval
        self._mod-=1
        if self._chaos<=0:
            row=(self._noise16()>>12)&0xF
            self.run_row(row)
            jitter=(self._noise16()&0xFF)-128
            fac=1.0+jitter/256.0
            self._chaos=max(4,int(self.chaos_interval*max(0.25,min(2.0,fac))))
        self._chaos-=1

    def next_sample(self):
        self.tick()
        self.update_meta()
        n=self.nasty()
        b=self.beaut()
        m=self.morph
        out=(1.0-m)*n+m*b
        out=math.tanh(out*1.5)/1.2
        return max(-1.0,min(1.0,out))

class LiveAudio:
    def __init__(self,vm):
        self.vm=vm
        self.stream=None
    def start(self):
        if not HAS_SD or self.stream is not None:
            return
        def cb(outdata,frames,time,status):
            for i in range(frames):
                outdata[i,0]=self.vm.next_sample()
        self.stream=sd.OutputStream(
            samplerate=self.vm.hardware_sample_rate,
            channels=1,callback=cb,blocksize=256)
        self.stream.start()
    def stop(self):
        if self.stream:
            self.stream.stop();self.stream.close();self.stream=None

def make_sokoban_level():
    rows,cols=16,16
    board=[["#"]*cols for _ in range(rows)]
    for r in range(1,rows-1):
        for c in range(1,cols-1):
            board[r][c]=" "
    for _ in range(random.randint(4,7)):
        h=random.randint(2,4);w=random.randint(3,6)
        r0=random.randint(2,rows-h-2)
        c0=random.randint(2,cols-w-2)
        for r in range(r0,r0+h):
            for c in range(c0,c0+w):
                if 6<=r<=9 and 6<=c<=9 and random.random()<0.4: continue
                board[r][c]="#"

    floor=[(r,c) for r in range(1,rows-1) for c in range(1,cols-1) if board[r][c]==" "]
    if len(floor)<20: return make_sokoban_level()
    num_goals=random.randint(3,5)
    goals=random.sample(floor,num_goals)
    for (r,c) in goals: board[r][c]="."
    floor=[fc for fc in floor if fc not in goals]
    boxes=random.sample(floor,num_goals)
    for (r,c) in boxes: board[r][c]="$"
    floor=[fc for fc in floor if fc not in boxes]

    def ok(rr,cc):
        if board[rr][cc]!=" ": return False
        n=0
        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            r2, c2=rr+dr,cc+dc
            if board[r2][c2] in (" ",".","$"): n+=1
        return n>=2

    random.shuffle(floor)
    for (rr,cc) in floor:
        if ok(rr,cc):
            board[rr][cc]="@"; return board
    rr,cc=floor[0]
    board[rr][cc]="@"
    return board

def find_player(board):
    for r in range(16):
        for c in range(16):
            if board[r][c] in ("@","+"): return r,c
    return 1,1

def move_player(board,pr,pc,dr,dc):
    nr, nc = pr+dr, pc+dc
    if not (0<=nr<16 and 0<=nc<16): return False,pr,pc,[]
    dest=board[nr][nc]
    changed=[]
    def set_tile(rr,cc,ch):
        if board[rr][cc]!=ch:
            board[rr][cc]=ch;changed.append((rr,cc))
    pog=(board[pr][pc]=="+")
    if dest=="#": return False,pr,pc,[]
    if dest in (" ","."):
        set_tile(pr,pc,"." if pog else " ")
        set_tile(nr,nc,"+" if dest=="." else "@")
        return True,nr,nc,changed
    if dest in ("$","*"):
        br,bc=nr+dr,nc+dc
        if not(0<=br<16 and 0<=bc<16): return False,pr,pc,[]
        beyond=board[br][bc]
        if beyond in (" ","."):
            set_tile(nr,nc,"+" if beyond=="." else "@")
            set_tile(br,bc,"*" if beyond=="." else "$")
            set_tile(pr,pc,"." if pog else " ")
            return True,nr,nc,changed
    return False,pr,pc,[]

def randomise_area_from(program,pulses,r,c,max_p=40):
    h=random.randint(2,5);w=random.randint(2,5)
    vals=list(OPCODES.values())
    for rr in range(r,min(16,r+h)):
        for cc in range(c,min(16,c+w)):
            program.grid[rr][cc]=random.choice(vals)
            pulses[rr][cc]=max_p

def apply_sokoban_changes(program,pulses,changed):
    for (r,c) in changed:
        randomise_area_from(program,pulses,r,c)

def example_program():
    p=BitGridProgram.empty()
    g=p.grid
    for c in range(16):
        g[0][c]=OPCODES["STEP"] if c%2==0 else OPCODES["NOISE"]
    for c in range(16):
        if c%4==0: g[1][c]=OPCODES["ADD"]; g[2][c]=OPCODES["ADD"]
        elif c%4==1: g[1][c]=OPCODES["SUB"]; g[2][c]=OPCODES["XOR"]
        elif c%4==2: g[1][c]=OPCODES["ROL"]; g[2][c]=OPCODES["AND"]
        else: g[1][c]=OPCODES["NOP"]; g[2][c]=OPCODES["NOP"]
    for c in range(16):
        g[3][c]=OPCODES["ROL"] if c%3==0 else OPCODES["XOR"]
        g[4][c]=OPCODES["NOISE"] if c in (0,5,9,13) else OPCODES["AND"]
        g[5][c]=OPCODES["MUL"] if c%2 else OPCODES["ADD"]
    for c in range(16):
        g[6][c]=OPCODES["ADD"] if c%3 else OPCODES["NOISE"]
        g[7][c]=OPCODES["ADD"] if c%2 else OPCODES["SUB"]
        g[10][c]=OPCODES["XOR"] if c%2 else OPCODES["ADD"]
        g[11][c]=OPCODES["ADD"] if c%5 else OPCODES["NOISE"]
        g[12][c]=OPCODES["ADD"] if c%2 else OPCODES["SUB"]
        g[14][c]=OPCODES["ADD"] if c%3 else OPCODES["SUB"]
    return p

def run_fullscreen_app():
    pygame.init()
    screen=pygame.display.set_mode((0,0),pygame.FULLSCREEN)
    info=pygame.display.Info()
    width,height=info.current_w,info.current_h
    clock=pygame.time.Clock()
    t=0.0

    program=example_program()
    vm=BitGridVM(program)
    audio=LiveAudio(vm); audio.start()

    board=make_sokoban_level()
    pr,pc=find_player(board)
    pulses=[[0]*16 for _ in range(16)]
    max_p=40

    grid=int(min(width,height)*0.9)
    cell=grid/16.0
    ox=(width-grid)/2.0
    oy=(height-grid)/2.0

    trail=pygame.Surface((width,height),pygame.SRCALPHA)
    bg_w=max(160,width//8)
    bg_h=max(90,height//8)
    bg=pygame.Surface((bg_w,bg_h))

    WALL=(10,10,40)
    FLOOR=(3,3,20)
    GOAL=(80,90,200)
    BOX=(200,130,80)
    BOXG=(230,200,120)
    PLY=(180,240,255)
    PLYG=(210,255,255)

    run=True
    while run:
        dt=clock.tick(60)/1000.0
        t+=dt
        for e in pygame.event.get():
            if e.type==pygame.QUIT: run=False
            elif e.type==pygame.KEYDOWN:
                if e.key==pygame.K_ESCAPE: run=False
                dr=dc=0
                if e.key==pygame.K_UP: dr=-1
                elif e.key==pygame.K_DOWN: dr=1
                elif e.key==pygame.K_LEFT: dc=-1
                elif e.key==pygame.K_RIGHT: dc=1
                if dr!=0 or dc!=0:
                    moved,prn,pcn,ch=move_player(board,pr,pc,dr,dc)
                    if moved:
                        pr,pc=prn,pcn
                        apply_sokoban_changes(program,pulses,ch)

        for r in range(16):
            for c in range(16):
                if pulses[r][c]>0: pulses[r][c]-=1

        trail.fill((0,0,0,40),special_flags=pygame.BLEND_RGBA_SUB)

        for y in range(bg_h):
            v=y/bg_h*2-1
            for x in range(bg_w):
                u=x/bg_w*2-1
                dx,dy=u,v
                rr=math.sqrt(dx*dx+dy*dy)+1e-6
                ang=math.atan2(dy,dx)
                w1=math.sin(3*rr - t*1.2 + 2*math.sin(ang*2+t*0.7))
                w2=math.sin(4*(u+v)+t*1.5)
                w3=math.sin(6*(u-v)-t*1.1)
                p=(w1+w2+w3)/3
                cr=0.5+0.5*math.sin(p*3+t*0.9)
                cg=0.5+0.5*math.sin(p*3+t*1.1+2.1+0.7*math.sin(ang*3+t*0.5))
                cb=0.5+0.5*math.sin(p*3+t*0.8+4.2+0.9*math.cos(rr*5-t*0.4))
                vig=math.exp(-rr*1.4)
                cr*=vig;cg*=vig;cb*=vig
                bg.set_at((x,y),(clamp(int(cr*255)),clamp(int(cg*255)),clamp(int(cb*255))))

        screen.blit(pygame.transform.smoothscale(bg,(width,height)),(0,0))
        gglow=0.5*(math.sin(t*0.4)+1)

        for r in range(16):
            for c in range(16):
                ch=board[r][c]
                x=int(ox+c*cell);y=int(oy+r*cell)
                rect=pygame.Rect(x,y,int(cell)+1,int(cell)+1)
                if ch=="#": base=WALL
                elif ch in (" ","@","+","$","*"): base=FLOOR
                elif ch==".": base=GOAL
                else: base=FLOOR

                op=program.grid[r][c]
                ocol=OPCODE_COLORS.get(op,(80,80,80))
                ocol=cycle_color(ocol,t+0.15*r+0.23*c,strength=0.6)
                pf=pulses[r][c]/max_p if pulses[r][c]>0 else 0.0
                wob=1.0+0.15*math.sin(t*1.8+(r*0.5+c*0.9))
                glow=0.4*gglow+0.6*pf
                ba=0.25+0.4*pf+0.25*gglow
                ba=max(0,min(1,ba))
                bmix=blend(base,ocol,ba)
                final=blend(bmix,(255,255,255),glow*0.35)
                screen.fill(final,rect)

                cx=x+cell/2;cy=y+cell/2
                rad=int(cell*0.28*wob*(1+0.3*pf))
                aura_alpha=clamp(int(80+100*glow))
                pygame.draw.circle(trail,(*final,aura_alpha),(int(cx),int(cy)),int(rad*1.2))

                if ch in (".","+","*"):
                    ring=int(rad*1.05)
                    col=blend(GOAL,(255,255,255),0.4+0.3*pf)
                    pygame.draw.circle(screen,col,(int(cx),int(cy)),ring,2)

                if ch in ("$", "*"):
                    inset=cell*0.18*(1-0.2*(pf+gglow))
                    inset=max(0,min(cell*0.4,inset))
                    box_rect=pygame.Rect(int(x+inset),int(y+inset),
                                         int(cell-2*inset),int(cell-2*inset))
                    color=BOX if ch=="$" else BOXG
                    pygame.draw.rect(screen,color,box_rect,border_radius=int(cell*0.18))
                    inner_in=inset+cell*0.04
                    inner_rect=pygame.Rect(int(x+inner_in),int(y+inner_in),
                                           int(cell-2*inner_in),int(cell-2*inner_in))
                    inner_col=blend(color,(255,255,255),0.35+0.25*pf)
                    pygame.draw.rect(screen, inner_col, inner_rect, border_radius=int(cell*0.15))

                if ch in ("@","+"):
                    pcol=PLY if ch=="@" else PLYG
                    orr=int(rad*1.05);mrr=int(rad*0.75);irr=int(rad*0.45)
                    pygame.draw.circle(screen,pcol,(int(cx),int(cy)),orr)
                    mid=blend(pcol,(255,255,255),0.4+0.2*gglow)
                    pygame.draw.circle(screen,mid,(int(cx),int(cy)),mrr)
                    core=blend(pcol,(255,255,255),0.6+0.3*pf)
                    pygame.draw.circle(screen,core,(int(cx),int(cy)),irr)

        screen.blit(trail,(0,0),special_flags=pygame.BLEND_RGBA_ADD)
        pygame.display.flip()

    audio.stop()
    pygame.quit()

if __name__=="__main__":
    run_fullscreen_app()
