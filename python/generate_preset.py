from adaptive_audacity_preset import *

profile = load_profile('speaker_profiles/marshall_emberton_iii.json')
freqs, levels = read_spectrum('../Комната М/Комната_М_spectrum.txt')
preset_frequencies, gains, diagnostics, _, _ = create_gain_curve([(freqs, levels)], profile)
preset = build_audacity_preset(profile, preset_frequencies, gains)

Path('../Marshall_Emberton_III_Dynamic_Preset.txt').write_text(preset + '\n', encoding='utf-8')
print('Пресет сохранён в Marshall_Emberton_III_Dynamic_Preset.txt')
print()
print(preset)
