"""Independent finite-population enumeration and exact probability identities."""
from fractions import Fraction as Q
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory
import json,random,re
import experiment as lab
O=sorted(lab.OUTCOMES)
EVENTS=[set(c) for k in range(5) for c in combinations(O,k)]


def rejects(call):
    try:call()
    except (ValueError,TypeError,ZeroDivisionError):return
    raise AssertionError('Invalid input accepted')


def tests():
    groups=[];table=lab.load_table();D={(1,0),(1,1)};F={(0,1),(1,1)}
    assert table.total==1000
    assert table.probability(D)==Q(1,50)
    assert table.conditional(F,D)==Q(9,10)
    assert table.conditional(D,F)==Q(9,58)
    assert table.conditional(D,lab.OUTCOMES-F)==Q(1,442)
    assert not table.independent(D,F)
    groups.append('main joint marginal posterior and negative-flag hand results')
    rng=random.Random(16016);pairs=0;conditionals=0;bayes=0
    for fixture in range(100):
        counts={key:rng.randrange(10) for key in O}
        if sum(counts.values())==0:counts[O[0]]=1
        t=lab.FiniteTable(counts)
        population=[key for key in O for _ in range(counts[key])]
        def direct(event):return Q(sum(point in event for point in population),len(population))
        for a in EVENTS:
            assert t.probability(a)==direct(a)
            assert t.probability(lab.OUTCOMES-a)==1-direct(a)
            for b in EVENTS:
                pairs+=1
                assert t.probability(a|b)==direct(a)+direct(b)-direct(a&b)
                assert t.independent(a,b)==(direct(a&b)==direct(a)*direct(b))
                if direct(b):
                    selected=[point for point in population if point in b]
                    oracle=Q(sum(point in a for point in selected),len(selected))
                    assert t.conditional(a,b)==oracle;conditionals+=1
                else:rejects(lambda:t.conditional(a,b))
        if t.probability(D)>0 and t.probability(lab.OUTCOMES-D)>0 and t.probability(F)>0:
            result=lab.bayes_flag(t.probability(D),t.conditional(F,D),t.conditional(F,lab.OUTCOMES-D))
            assert result['posterior_defect_given_flag']==direct(D&F)/direct(F);bayes+=1
    groups.append('100 enumerated populations and all 256 event pairs per population')
    raw={key:1 for key in O};t=lab.FiniteTable(raw);raw[O[0]]=1000
    assert t.total==4 and t.probability({O[0]})==Q(1,4)
    assert t.probability([O[0],O[0]])==Q(1,4)
    assert lab.bayes_flag('1/10','9/10','1/10')['posterior_defect_given_flag']==Q(1,2)
    assert lab.bayes_flag('1/2','9/10','1/10')['posterior_defect_given_flag']==Q(9,10)
    assert lab.bayes_flag('1/1000','9/10','1/10')['posterior_defect_given_flag']==Q(1,112)
    groups.append('copied count ownership, duplicate event membership and exact prior scan')
    rep=lab.replacement_example()
    assert rep['with_replacement']=={'ordered_pairs':25,'both_defective':Q(4,25),'second_given_first_defective':Q(2,5)}
    assert rep['without_replacement']=={'ordered_pairs':20,'both_defective':Q(1,10),'second_given_first_defective':Q(1,4)}
    xor=lab.xor_world();assert all(xor['pairwise_independent'].values());assert xor['triple_111']==0 and xor['product_of_three_marginals']==Q(1,8)
    groups.append('all ordered draws and pairwise versus joint independence')
    obs1=lab.causal_world('chain');obs2=lab.causal_world('common_cause');assert obs1==obs2
    for fixed in (0,1):
        chain=lab.causal_world('chain',fixed);common=lab.causal_world('common_cause',fixed)
        assert sum(r['mass'] for r in chain if r['y']==1)==fixed
        assert sum(r['mass'] for r in common if r['y']==1)==Q(1,2)
    groups.append('two observationally identical mechanisms under both interventions')
    failures=[]
    for counts in [{}, {(0,0):1}, {k:0 for k in O}, {**raw,(0,0):-1}, {**raw,(0,0):True}, {**raw,(0,0):.5}, {(True,0):1,(0,0):1,(0,1):1,(1,1):1}]:failures.append(lambda counts=counts:lab.FiniteTable(counts))
    for event in [[(2,0)],[(True,0)],[[0]],'00',None]:failures.append(lambda event=event:table.probability(event))
    failures += [lambda:table.conditional(D,set()),lambda:lab.bayes_flag('0','1','0'),lambda:lab.bayes_flag('1','1','0'),lambda:lab.bayes_flag('1/2','0','0')]
    for p in [True,.1,'-1/10','11/10','nan','1/0',None]:failures.append(lambda p=p:lab.bayes_flag(p,'9/10','1/10'))
    failures += [lambda:lab.causal_world('other'),lambda:lab.causal_world('chain',True),lambda:lab.causal_world('chain',2)]
    for f in failures:rejects(f)
    groups.append(f'{len(failures)} invalid table event probability and intervention rejections')
    with TemporaryDirectory() as temp:
        a,b=Path(temp)/'a',Path(temp)/'b';lab.run(a);lab.run(b)
        for f in a.iterdir():assert f.read_bytes()==(b/f.name).read_bytes()
        original=lab.BASE;root=Path(temp)/'bad';(root/'data').mkdir(parents=True)
        malformed=['defective,flagged,count\n0,0,1,extra\n','defective,flagged,count\n0,0,1\n0,0,2\n','defective,flagged,count\n0,0,1.5\n']
        try:
            lab.BASE=root
            for content in malformed:
                (root/'data/inspection_counts.csv').write_text(content)
                rejects(lambda:lab.run(root/'outputs'));assert not (root/'outputs').exists()
        finally:lab.BASE=original
    groups.append('byte-identical reruns and three malformed-copy CSV failures before output creation')
    ns={};snippets=0
    for name in ['lecture.md','lab.md','answers.md']:
        if not (lab.BASE/name).exists():continue
        for code in re.findall(r'```python\n(.*?)```',(lab.BASE/name).read_text(),re.S):exec(code,ns);snippets+=1
    groups.append(f'{snippets} document Python snippets execute in order')
    return dict(status='passed',groups=len(groups),details=groups,population_fixtures=100,event_pairs=pairs,conditional_comparisons=conditionals,bayes_comparisons=bayes,rejected_cases=len(failures),malformed_csv_cases=3,python_snippets=snippets)
if __name__=='__main__':print(json.dumps(tests(),ensure_ascii=False,indent=2))
