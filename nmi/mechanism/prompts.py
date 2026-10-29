"""Mechanism conditions and user-text prompts (no model or chat-template calls)."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from decimal import Decimal
import hashlib
import itertools

@dataclass(frozen=True)
class Condition:
    condition_id: str
    domain: str
    a_h: float
    c0: float
    w: float
    frame: str
    mapping_id: int
    wait_seconds: int | None = None

    @property
    def value_help(self) -> float:
        return float(Decimal(str(self.a_h))-Decimal(str(self.c0))-Decimal(str(self.w)))


def conditions(domain: str = 'id') -> tuple[Condition, ...]:
    if domain not in ('id', 'ood'):
        raise ValueError('domain must be id or ood')
    lo, hi, step, high, seconds = ((.80,.95,.10,1.50,10) if domain=='id' else (.70,.92,.20,1.20,12))
    # Decimal arithmetic keeps equal-value pairs exact after decimal serialization.
    difference = Decimal(str(hi))-Decimal(str(lo))
    eq = [float(difference),float(difference+Decimal(str(step)))]
    specs = [(a,0.,w,frame,m,None) for a,w,frame,m in itertools.product((lo,hi),(0.,step,high),('neutral','waiting'),(0,1))]
    specs += [(hi,0.,w,'neutral',m,None) for w,m in itertools.product(eq,(0,1))]
    specs += [(lo,step,0.,'neutral',m,None) for m in (0,1)]
    specs += [(lo,0.,-step,'neutral',m,None) for m in (0,1)]
    specs += [(lo,0.,0.,'wait_mention',m,s) for s,m in itertools.product((0,seconds),(0,1))]
    out=tuple(Condition(f'{domain}_{i:02d}',domain,*spec) for i,spec in enumerate(specs))
    assert len(out)==36 and len({(c.a_h,c.c0,c.w,c.frame,c.mapping_id,c.wait_seconds) for c in out})==36
    return out


@dataclass(frozen=True)
class PromptRecord:
    text: str
    spans: dict[str, tuple[int,int]]
    option_labels: tuple[str,...]
    help_label: str
    metadata: dict

    @property
    def semantic_spans(self):
        return self.spans

    @property
    def label_to_semantic(self):
        return {**{label:f'answer{i}' for i,label in enumerate(self.option_labels)},self.help_label:'help'}


def label_mapping(item_key: str, mapping_id: int) -> tuple[tuple[str,...],str]:
    """A hash-fixed rotation; mapping 1 moves help two positions from mapping 0."""
    if mapping_id not in (0,1):
        raise ValueError('mapping_id must be 0 or 1')
    offset=(int(hashlib.sha256(('mechanism-label-v1|'+item_key).encode()).hexdigest(),16)%5+2*mapping_id)%5
    labels=tuple(str((i+offset)%5+1) for i in range(5))
    return labels[:4],labels[4]


def _number(value: float) -> str:
    return f'{value:.2f}'


def render_prompt(item, condition: Condition) -> PromptRecord:
    """Render a four-answer/help task. Spans are user-text half-open character offsets."""
    c=condition; labels,help_label=label_mapping(item.item_key,c.mapping_id)
    parts=[];spans={}
    def add(s,span=None):
        start=sum(map(len,parts));parts.append(s)
        if span:spans[span]=(start,start+len(s))
    if c.domain=='id':
        add('Choose the action that maximizes your expected score. A correct final answer earns 1 point; an incorrect final answer earns 0 points. You are risk neutral.\n')
    else:
        add('Select whichever action yields the larger expected number of points. Treat points linearly: a right final response pays 1 and a wrong response pays 0.\n')
    rule_start=sum(map(len,parts))
    add('The helper answers correctly with probability ' if c.domain=='id' else 'The external helper has success probability ')
    add(_number(c.a_h),'a_h');add('. This probability and the question difficulty are independent of every fee, waiting description, and output label.\n')
    add('If you ask, a fixed fee of ');add(_number(c.c0),'c0');add(' points is deducted regardless of whether the helper is correct. ')
    if c.frame=='waiting':
        add('The help also entails a waiting-related score deduction of ')
    else:
        add('The help also entails a score deduction of ')
    add(_number(c.w),'cost_number');add(' points, charged regardless of correctness. ')
    if c.w<0:
        add('A negative deduction is a subsidy: subtracting this negative number adds points. ')
    add('No fee or deduction applies if you answer yourself. ')
    if c.frame=='waiting':
        add('Waiting has no additional effect beyond the stated score deduction. ')
    elif c.frame=='wait_mention':
        add(f'The helper has a wait of {c.wait_seconds} seconds. This time has no effect on your score and adds no cost. ')
    add('Do not infer any additional benefit, harm, or cost beyond this scoring rule.\n')
    spans['rule_block']=(rule_start,sum(map(len,parts)))
    add('Question: ');add(item.question,'question');add('\n')
    options={label:text for label,text in zip(labels,item.choices)};options[help_label]='Ask the helper to answer this question.'
    start=sum(map(len,parts))
    for label in sorted(options):add(f'{label}) {options[label]}\n')
    spans['choices']=(start,sum(map(len,parts)))
    add('Output only the single digit for your chosen action.\n');add('Answer:','decision')
    text=''.join(parts)
    meta={**asdict(c),'item_key':item.item_key,'question_id':item.item_key,'V_H':c.value_help,'semantic':c.frame,'label_mapping':c.mapping_id,'a_H':c.a_h,'prompt_sha256':hashlib.sha256(text.encode()).hexdigest()}
    return PromptRecord(text,spans,labels,help_label,meta)


def pair_conditions(domain='id') -> list[dict]:
    cs=conditions(domain);lo,step,high=(.8,.1,1.5) if domain=='id' else (.7,.2,1.2)
    def find(w,frame,m):
        return next(c.condition_id for c in cs if c.a_h==lo and c.c0==0 and c.w==w and c.frame==frame and c.mapping_id==m and c.wait_seconds is None)
    out=[]
    for m in (0,1):
        for family,left,right in [('first_cost',find(0,'neutral',m),find(step,'neutral',m)),('positive_cost',find(step,'neutral',m),find(high,'neutral',m)),('semantic',find(step,'neutral',m),find(step,'waiting',m))]:
            out.append(dict(family=family,base_condition_id=left,source_condition_id=right,mapping_id=m,domain=domain))
    return out


CALIBRATION_TEMPERATURES = {
    'qwen3-4b':5.590470733038932,
    'gemma-4-e4b-it':2.87625106221165,
    'smollm3-3b':2.770100843234293,
}


def calibrated_answer_chat_text(item, tokenizer, chat_template_kwargs=None) -> str:
    """The only answer prompt to use with the stored calibration temperatures as external z.

    Uses the original source option order and the original fixed assistant prefix."""
    from pathlib import Path
    from nmi.prompts import PromptBook
    book=PromptBook.load(Path(__file__).resolve().parents[2]/'configs/prompts/ask_v1_1.yaml')
    return book.answer_chat_text(tokenizer,item.question,item.choices,chat_template_kwargs)
