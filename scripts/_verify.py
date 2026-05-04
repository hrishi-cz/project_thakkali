"""Verify all audit items are applied."""
text = open(r'frontend\app_enhanced.py', encoding='utf-8').read()
checks = [
    ('CSS #0b0b1a', '--bg:#0b0b1a' in text),
    ('Hero autovision-hero', 'autovision-hero' in text),
    ('Stepper av-stepper', 'av-stepper' in text),
    ('_kv_table helper', 'def _kv_table' in text),
    ('_model_card helper', 'def _model_card' in text),
    ('_paper_cite helper', 'def _paper_cite' in text),
    ('Research footer', 'av-footer-cite' in text),
    ('Gradient buttons', 'linear-gradient(135deg,var(--violet)' in text),
    ('pulse-glow anim', 'pulse-glow' in text),
    ('model-card CSS', '.model-card' in text),
    ('paper-cite CSS', '.paper-cite' in text),
    ('JetBrains Mono', 'JetBrains Mono' in text),
    ('av-card CSS', '.av-card' in text),
    ('av-alert CSS', '.av-alert' in text),
    ('amber-cta CSS', 'amber-cta' in text),
    ('badge.red CSS', '.badge.red' in text or 'badge.red' in text),
    ('input focus CSS', 'textarea:focus' in text),
    ('secondary btn', 'secondary' in text),
    ('global-target wired', 'global-target' in text),
    ('fit-analysis wired', 'fit-analysis' in text),
    ('drift-status wired', 'drift-status' in text),
]
missing = []
for name, ok in checks:
    status = 'OK' if ok else 'MISS'
    print(f'{status}: {name}')
    if not ok:
        missing.append(name)

print(f'\nst.json count: {text.count("st.json(")}')
print(f'\nMissing items: {len(missing)}')
for m in missing:
    print(f'  - {m}')
