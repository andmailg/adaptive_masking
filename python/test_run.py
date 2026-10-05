from adaptive_audacity_preset import *

profile = load_profile('speaker_profiles/marshall_emberton_iii.json')
freqs, levels = read_spectrum('../Комната М/Комната_М_spectrum.txt')
preset_frequencies, gains, diagnostics, raw_prominences, limited_prominences = create_gain_curve([(freqs, levels)], profile)

print('Gains:', [f'{g:.2f}' for g in gains])
print()
for d in diagnostics:
    print(f'{d["frequency"]:>7.0f} Hz | raw_gain={d["raw_gain"]:>7.2f} | prominence={d["limited_prominence"]:>7.2f} | speaker={d["speaker_response"]:>7.2f} | masking={d["masking_level"]:>7.2f} | final={d["final_gain"]:>7.2f}')
