use crate::{fixed, menksoft, mongol};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Location {
    Isol,
    Init,
    Medi,
    Fina,
}

pub fn to_menksoft(input: &str) -> String {
    let codepoints: Vec<u32> = input.chars().map(|c| c as u32).collect();
    let mut output = Vec::new();
    let mut index = 0;

    while index < codepoints.len() {
        if !is_mongolian_segment_character(codepoints[index]) {
            output.push(map_punctuation(codepoints[index]).unwrap_or(codepoints[index]));
            index += 1;
            continue;
        }

        let start = index;
        while index < codepoints.len() && is_mongolian_segment_character(codepoints[index]) {
            index += 1;
        }
        output.extend(process_segment(&codepoints[start..index]));
    }

    from_codepoints(&output)
}

pub fn to_nominal_unicode(input: &str) -> String {
    input
        .chars()
        .filter_map(|c| {
            let cp = c as u32;
            match cp {
                mongol::NNBS => char::from_u32(mongol::MVS),
                cp if mongol::is_fvs(cp) => None,
                cp if mongol::is_zero_width_noise(cp) => None,
                _ => Some(c),
            }
        })
        .collect()
}

pub fn clean_mw_unicode(input: &str) -> String {
    input
        .chars()
        .filter_map(|c| {
            let cp = c as u32;
            match cp {
                mongol::NNBS => char::from_u32(mongol::MVS),
                cp if mongol::is_zero_width_noise(cp) => None,
                _ => Some(c),
            }
        })
        .collect()
}

fn process_segment(segment: &[u32]) -> Vec<u32> {
    if segment.iter().any(|&cp| is_todo_sibe_manchu(cp)) {
        return segment.to_vec();
    }

    let normalized: Vec<u32> = segment
        .iter()
        .map(|&cp| match cp {
            mongol::NNBS => mongol::MVS,
            mongol::ZWJ => mongol::NIRUGU,
            cp if mongol::is_zero_width_noise(cp) => 0,
            _ => cp,
        })
        .filter(|&cp| cp != 0)
        .collect();

    split_by_mvs_words(&normalized)
        .into_iter()
        .flat_map(|word| process_word(&word))
        .collect()
}

fn process_word(word: &[u32]) -> Vec<u32> {
    if let Some(mapped) = fixed::lookup_unicode_word(word) {
        return mapped.to_vec();
    }

    convert_word_contextually(word)
}

fn split_by_mvs_words(segment: &[u32]) -> Vec<Vec<u32>> {
    let mut words = Vec::new();
    let mut word = Vec::new();
    let mut index = 0;

    while index < segment.len() {
        let cp = segment[index];
        if cp == mongol::MVS {
            if is_mvs_ae_word(index, segment) {
                word.push(cp);
                word.push(segment[index + 1]);
                words.push(word);
                word = Vec::new();
                index += 2;
            } else {
                if !word.is_empty() {
                    words.push(word);
                    word = Vec::new();
                }
                word.push(cp);
                index += 1;
            }
        } else {
            word.push(cp);
            index += 1;
        }
    }

    if !word.is_empty() {
        words.push(word);
    }

    words
}

fn is_mvs_ae_word(index: usize, segment: &[u32]) -> bool {
    let Some(&next) = segment.get(index + 1) else {
        return false;
    };
    matches!(next, mongol::A | mongol::E)
        && segment.get(index + 2).is_none_or(|&cp| cp == mongol::MVS)
}

fn convert_word_contextually(word: &[u32]) -> Vec<u32> {
    let word = strip_glide_ya(word);
    let word = word.as_slice();
    let base_indices: Vec<usize> = word
        .iter()
        .enumerate()
        .filter_map(|(index, &cp)| (!mongol::is_control(cp)).then_some(index))
        .collect();
    if base_indices.is_empty() {
        // A bare separator word (corpus noise): keep the suffix gap visible
        // instead of silently dropping it.
        return word
            .iter()
            .filter(|&&cp| cp == mongol::MVS)
            .map(|_| 0xE263)
            .collect();
    }

    let first = *base_indices.first().unwrap();
    let last = *base_indices.last().unwrap();
    let mut output = Vec::new();
    let mut prev_base: Option<usize> = None;

    for &index in &base_indices {
        let cp = word[index];
        let location = if first == last {
            Location::Isol
        } else if index == first {
            Location::Init
        } else if index == last {
            Location::Fina
        } else {
            Location::Medi
        };
        let above = previous_base(word, index);
        let below = next_base(word, index);
        let fvs = word
            .get(index + 1)
            .copied()
            .filter(|&next| mongol::is_fvs(next))
            .unwrap_or(0);

        // base_indices treats MVS (inside the is_control range) as a control
        // and drops it, which used to swallow the suffix separator of every
        // word the fixed table does not list (᠊ᠪᠠᠨ, ᠊ᠲᠠᠢ, …). Re-emit it as
        // the Menksoft gap E263 — except before a separated final vowel,
        // whose post-MVS glyphs (E26A/E274/…) carry the gap themselves.
        let gap_lo = prev_base.map_or(0, |p| p + 1);
        if word[gap_lo..index].contains(&mongol::MVS)
            && !(matches!(cp, mongol::A | mongol::E) && index == last)
        {
            output.push(0xE263);
        }

        output.push(map_letter(word, index, cp, location, above, below, fvs));
        prev_base = Some(index);
    }

    output
}

fn map_letter(
    word: &[u32],
    index: usize,
    cp: u32,
    location: Location,
    above: u32,
    below: u32,
    fvs: u32,
) -> u32 {
    // Right after the suffix separator a letter takes its dedicated
    // suffix-initial form where Menksoft defines one (GB/T fixed sequences:
    // ᠊ᠲ E309, ᠊ᠳ E310, ᠊ᠶ E321, ᠊ᠤ E292, ᠊ᠦ E2AC, single ᠊ᠢ E282); letters
    // without one (ᠪ …) keep their regular positional form. ᠠ/ᠡ are handled
    // by their own post-MVS branches below.
    if above == mongol::MVS && fvs == 0 {
        match cp {
            mongol::TA => return 0xE309,
            mongol::DA => return 0xE310,
            mongol::YA => return 0xE321,
            mongol::U => return 0xE292,
            mongol::UE => return 0xE2AC,
            mongol::I if below == 0 => return 0xE282,
            // Suffix-initial ᠠ/ᠡ stay headless (the separated forms the
            // fixed ᠊ᠠᠴᠠ sequence uses) — the regular word-initial forms
            // carry a crown and would read as a fresh word.
            mongol::A => return 0xE26A,
            mongol::E => return 0xE274,
            _ => {}
        }
    }
    match cp {
        mongol::A => match location {
            Location::Isol => match fvs {
                mongol::FVS1 => 0xE265,
                mongol::FVS2 => 0xE26A,
                _ => 0xE264,
            },
            Location::Init => {
                if fvs == mongol::FVS1 {
                    0xE267
                } else {
                    0xE266
                }
            }
            Location::Medi => {
                if fvs == mongol::FVS1 {
                    0xE26E
                } else if below == mongol::MVS {
                    if is_round_letter(above) {
                        0xE26B
                    } else {
                        0xE268
                    }
                } else if is_round_letter(above) {
                    0xE26D
                } else {
                    0xE26C
                }
            }
            Location::Fina => {
                if fvs == mongol::FVS1 {
                    0xE269
                } else if above == mongol::MVS {
                    0xE26A
                } else if is_round_letter(above) {
                    0xE26B
                } else {
                    0xE268
                }
            }
        },
        mongol::E => match location {
            Location::Isol => {
                if fvs == mongol::FVS1 {
                    0xE274
                } else {
                    0xE270
                }
            }
            Location::Init => {
                if fvs == mongol::FVS1 {
                    0xE272
                } else {
                    0xE271
                }
            }
            Location::Medi => {
                if below == mongol::MVS {
                    if is_round_letter(above) {
                        0xE275
                    } else {
                        0xE273
                    }
                } else if is_round_letter_including_qg(above) {
                    0xE277
                } else {
                    0xE276
                }
            }
            Location::Fina => {
                if fvs == mongol::FVS1 {
                    0xE269
                } else if above == mongol::MVS {
                    0xE274
                } else if is_round_letter_including_qg(above) {
                    0xE275
                } else {
                    0xE273
                }
            }
        },
        mongol::I => match location {
            Location::Isol => {
                if fvs == mongol::FVS1 {
                    0xE282
                } else {
                    0xE279
                }
            }
            Location::Init => {
                if fvs == mongol::FVS1 {
                    0xE280
                } else {
                    0xE27A
                }
            }
            Location::Medi => {
                if fvs == mongol::FVS1 {
                    0xE27D
                } else if fvs == mongol::FVS2
                    || context_calls_for_double_tooth_i(word, index, above, below)
                {
                    0xE281
                } else if below == mongol::MVS {
                    0xE27B
                } else if is_round_letter_including_qg(above) {
                    0xE27F
                } else {
                    0xE27E
                }
            }
            Location::Fina => {
                if is_round_letter_including_qg(above) {
                    0xE27C
                } else {
                    0xE27B
                }
            }
        },
        mongol::O => map_ou(
            location, above, below, fvs, 0xE283, 0xE284, 0xE289, 0xE28A, 0xE288, 0xE285, 0xE286,
            0xE287,
        ),
        mongol::U => map_ou(
            location, above, below, fvs, 0xE28B, 0xE28C, 0xE291, 0xE292, 0xE290, 0xE28D, 0xE28E,
            0xE28F,
        ),
        mongol::OE => map_oe_ue(word, index, location, above, below, fvs, false),
        mongol::UE => map_oe_ue(word, index, location, above, below, fvs, true),
        mongol::EE => match location {
            Location::Isol => 0xE2AD,
            Location::Init => 0xE2AE,
            Location::Medi => 0xE2B0,
            Location::Fina => 0xE2AF,
        },
        mongol::NA => map_na(location, below, fvs),
        mongol::ANG => map_ang(location, below),
        mongol::BA => map_bpfkkh(
            location, below, fvs, 0xE2C1, 0xE2C2, 0xE2C7, 0xE2C5, 0xE2C6, 0xE2C7, 0xE2C3, 0xE2C4,
        ),
        mongol::PA => map_bpfkkh(
            location, below, fvs, 0xE2C8, 0xE2C9, 0xE2CD, 0xE2CB, 0xE2CC, 0xE2CD, 0xE2CA, 0xE2CA,
        ),
        mongol::QA => map_qg(location, below, fvs, false),
        mongol::GA => map_qg(location, below, fvs, true),
        mongol::MA => map_mala(
            location, above, below, 0xE2F2, 0xE2F1, 0xE2F2, 0xE2F4, 0xE2F5, 0xE2F6, 0xE2F3,
        ),
        mongol::LA => map_mala(
            location, above, below, 0xE2F8, 0xE2F7, 0xE2F8, 0xE2FA, 0xE2FB, 0xE2FC, 0xE2F9,
        ),
        mongol::SA => map_simple_consonant(
            location, below, fvs, 0xE2FE, 0xE2FD, 0xE2FE, 0xE301, 0xE302, 0xE2FF, 0xE300,
        ),
        mongol::SHA => map_simple_consonant(
            location, below, fvs, 0xE304, 0xE303, 0xE304, 0xE306, 0xE307, 0xE305, 0xE305,
        ),
        mongol::TA => match location {
            Location::Isol => 0xE309,
            Location::Init => {
                if wants_tooth(below) {
                    0xE308
                } else {
                    0xE309
                }
            }
            Location::Medi => {
                if fvs == mongol::FVS1 && wants_tooth(below) {
                    0xE30C
                } else if fvs == mongol::FVS1 {
                    0xE30D
                } else {
                    0xE30B
                }
            }
            Location::Fina => 0xE30A,
        },
        mongol::DA => match location {
            Location::Isol => {
                if fvs == mongol::FVS1 {
                    0xE30F
                } else {
                    0xE310
                }
            }
            Location::Init => {
                if fvs == mongol::FVS1 {
                    0xE310
                } else if wants_tooth(below) {
                    0xE30E
                } else {
                    0xE30F
                }
            }
            Location::Medi => {
                if fvs == mongol::FVS1 {
                    0xE313
                } else {
                    0xE314
                }
            }
            Location::Fina => {
                if fvs == mongol::FVS1 {
                    0xE312
                } else {
                    0xE311
                }
            }
        },
        mongol::CHA => map_four(location, 0xE315, 0xE315, 0xE317, 0xE316),
        mongol::JA => match location {
            Location::Isol => {
                if fvs == mongol::FVS1 {
                    0xE31C
                } else {
                    0xE318
                }
            }
            Location::Init => {
                if wants_tooth(below) {
                    0xE319
                } else {
                    0xE31A
                }
            }
            Location::Medi => {
                if below == mongol::MVS {
                    0xE31C
                } else {
                    0xE31D
                }
            }
            Location::Fina => {
                if fvs == mongol::FVS1 {
                    0xE31C
                } else {
                    0xE31B
                }
            }
        },
        mongol::YA => match location {
            Location::Isol => {
                if fvs == mongol::FVS1 {
                    0xE321
                } else {
                    0xE31E
                }
            }
            Location::Init => {
                if fvs == mongol::FVS1 {
                    0xE321
                } else {
                    0xE31E
                }
            }
            Location::Medi => {
                if fvs == mongol::FVS1 {
                    0xE321
                } else if fvs == mongol::FVS2 {
                    0xE281
                } else if below == mongol::MVS {
                    0xE31F
                } else {
                    0xE320
                }
            }
            Location::Fina => 0xE31F,
        },
        mongol::RA => match location {
            Location::Isol => 0xE322,
            Location::Init => {
                if wants_tooth(below) {
                    0xE323
                } else {
                    0xE322
                }
            }
            Location::Medi => {
                if below == mongol::MVS {
                    0xE325
                } else if wants_tooth(below) {
                    0xE327
                } else {
                    0xE326
                }
            }
            Location::Fina => 0xE325,
        },
        mongol::WA => match location {
            Location::Isol | Location::Init => 0xE329,
            Location::Medi => {
                if fvs == mongol::FVS1 {
                    0xE286
                } else if below == mongol::MVS {
                    0xE32B
                } else {
                    0xE32C
                }
            }
            Location::Fina => {
                if fvs == mongol::FVS1 {
                    0xE32B
                } else {
                    0xE32A
                }
            }
        },
        mongol::FA => map_bpfkkh(
            location, below, fvs, 0xE32D, 0xE32E, 0xE332, 0xE330, 0xE331, 0xE332, 0xE32F, 0xE32F,
        ),
        mongol::KA => map_bpfkkh(
            location, below, fvs, 0xE333, 0xE334, 0xE333, 0xE336, 0xE337, 0xE338, 0xE335, 0xE335,
        ),
        mongol::KHA => map_bpfkkh(
            location, below, fvs, 0xE339, 0xE33A, 0xE339, 0xE33C, 0xE33D, 0xE33E, 0xE33B, 0xE33B,
        ),
        mongol::TSA => map_four(location, 0xE33F, 0xE33F, 0xE341, 0xE340),
        mongol::ZA => map_four(location, 0xE342, 0xE342, 0xE344, 0xE343),
        mongol::HAA => map_four(location, 0xE345, 0xE345, 0xE347, 0xE346),
        mongol::ZRA => map_four(location, 0xE348, 0xE348, 0xE349, 0xE34A),
        mongol::LHA => match location {
            Location::Isol | Location::Init => 0xE34B,
            Location::Medi | Location::Fina => {
                if is_round_letter(above) || above == mongol::ANG {
                    0xE34D
                } else {
                    0xE34C
                }
            }
        },
        mongol::ZHI => 0xE34E,
        mongol::CHI => 0xE34F,
        mongol::MVS | mongol::NNBS => menksoft::NONBREAKING_SPACE,
        mongol::NIRUGU | mongol::ZWJ => 0xE23E,
        mongol::BIRGA => 0xE234,
        _ => map_punctuation(cp).unwrap_or(cp),
    }
}

fn map_ou(
    location: Location,
    above: u32,
    below: u32,
    fvs: u32,
    isol: u32,
    init: u32,
    medi: u32,
    medi_bp: u32,
    medi_fvs1: u32,
    fina: u32,
    fina_fvs1: u32,
    fina_fvs1_bp: u32,
) -> u32 {
    match location {
        Location::Isol => isol,
        Location::Init => init,
        Location::Medi => {
            if fvs == mongol::FVS1 {
                medi_fvs1
            } else if below == mongol::MVS {
                if is_round_letter(above) {
                    fina_fvs1_bp
                } else {
                    fina
                }
            } else if is_round_letter(above) {
                medi_bp
            } else {
                medi
            }
        }
        Location::Fina => {
            if fvs == mongol::FVS1 {
                if is_round_letter(above) {
                    fina_fvs1_bp
                } else {
                    fina_fvs1
                }
            } else {
                if is_round_letter(above) {
                    fina_fvs1_bp
                } else {
                    fina
                }
            }
        }
    }
}

fn map_oe_ue(
    word: &[u32],
    index: usize,
    location: Location,
    above: u32,
    below: u32,
    fvs: u32,
    is_ue: bool,
) -> u32 {
    let (
        isol,
        isol_fvs1,
        isol_fvs2,
        init,
        init_fvs1,
        medi,
        medi_bp,
        medi_fvs1,
        medi_fvs1_bp,
        medi_fvs2,
        fina,
        fina_bp,
        fina_fvs1,
        fina_fvs1_bp,
        fina_fvs2,
        fina_fvs2_bp,
    ) = if is_ue {
        (
            0xE2A0, 0xE2A1, 0xE2A3, 0xE2A2, 0xE2AC, 0xE2AB, 0xE2AC, 0xE2A9, 0xE2AA, 0xE2A8, 0xE2A3,
            0xE2A7, 0xE2A4, 0xE2A5, 0xE2A6, 0xE2A7,
        )
    } else {
        (
            0xE293, 0xE294, 0xE293, 0xE295, 0xE295, 0xE29E, 0xE29F, 0xE29C, 0xE29D, 0xE29B, 0xE296,
            0xE29A, 0xE297, 0xE298, 0xE299, 0xE29A,
        )
    };

    match location {
        Location::Isol => match fvs {
            mongol::FVS1 => isol_fvs1,
            mongol::FVS2 => isol_fvs2,
            _ => isol,
        },
        Location::Init => {
            if fvs == mongol::FVS1 {
                init_fvs1
            } else {
                init
            }
        }
        Location::Medi => {
            if fvs == mongol::FVS1 || mongol::needs_long_tooth_u(word, index) {
                if is_round_letter_including_qg(above) {
                    medi_fvs1_bp
                } else {
                    medi_fvs1
                }
            } else if fvs == mongol::FVS2 {
                medi_fvs2
            } else if below == mongol::MVS {
                if is_round_letter(above) {
                    fina_fvs2_bp
                } else {
                    fina
                }
            } else if is_round_letter_including_qg(above) {
                medi_bp
            } else {
                medi
            }
        }
        Location::Fina => match fvs {
            mongol::FVS1 => {
                if is_round_letter_including_qg(above) {
                    fina_fvs1_bp
                } else {
                    fina_fvs1
                }
            }
            mongol::FVS2 => {
                if is_round_letter_including_qg(above) {
                    fina_fvs2_bp
                } else {
                    fina_fvs2
                }
            }
            _ => {
                if is_round_letter_including_qg(above) {
                    fina_bp
                } else {
                    fina
                }
            }
        },
    }
}

fn map_na(location: Location, below: u32, fvs: u32) -> u32 {
    match location {
        Location::Isol => {
            if fvs == mongol::FVS1 {
                0xE2B4
            } else {
                0xE2B3
            }
        }
        Location::Init => {
            if fvs == mongol::FVS1 {
                if wants_tooth(below) {
                    0xE2B2
                } else {
                    0xE2B4
                }
            } else if wants_tooth(below) {
                0xE2B1
            } else {
                0xE2B3
            }
        }
        Location::Medi => {
            if below == mongol::GA || below == mongol::QA {
                if fvs == mongol::FVS1 {
                    0xE2BF
                } else {
                    0xE2C0
                }
            } else if below == mongol::MVS {
                if fvs == mongol::FVS1 {
                    0xE2B6
                } else {
                    0xE2B5
                }
            } else if fvs == mongol::FVS1 {
                if wants_tooth(below) {
                    0xE2B7
                } else {
                    0xE2B9
                }
            } else if wants_tooth(below) {
                0xE2B8
            } else {
                0xE2BA
            }
        }
        Location::Fina => {
            if fvs == mongol::FVS1 {
                0xE2B6
            } else {
                0xE2B5
            }
        }
    }
}

fn map_ang(location: Location, below: u32) -> u32 {
    match location {
        Location::Isol | Location::Init => {
            if matches!(below, mongol::QA | mongol::GA) {
                0xE2BC
            } else {
                choose_tooth_round_stem(below, 0xE2BC, 0xE2BD, 0xE2BE)
            }
        }
        Location::Medi => {
            if matches!(below, mongol::QA | mongol::GA) {
                0xE2BC
            } else {
                choose_tooth_round_stem(below, 0xE2BC, 0xE2BD, 0xE2BE)
            }
        }
        Location::Fina => 0xE2BB,
    }
}

fn map_bpfkkh(
    location: Location,
    below: u32,
    fvs: u32,
    isol_or_init_tooth: u32,
    init_ou: u32,
    init_stem: u32,
    medi_tooth: u32,
    medi_ou: u32,
    medi_stem: u32,
    fina: u32,
    fina_fvs1: u32,
) -> u32 {
    match location {
        Location::Isol => isol_or_init_tooth,
        Location::Init => {
            if is_ou_vowel(below) {
                init_ou
            } else if wants_tooth(below) {
                isol_or_init_tooth
            } else {
                init_stem
            }
        }
        Location::Medi => {
            if is_ou_vowel(below) {
                medi_ou
            } else if wants_tooth(below) {
                medi_tooth
            } else {
                medi_stem
            }
        }
        Location::Fina => {
            if fvs == mongol::FVS1 {
                fina_fvs1
            } else {
                fina
            }
        }
    }
}

fn map_qg(location: Location, below: u32, fvs: u32, is_ga: bool) -> u32 {
    if is_ga {
        return match location {
            Location::Isol => match fvs {
                mongol::FVS1 => 0xE2E5,
                mongol::FVS2 => 0xE2E3,
                mongol::FVS4 => 0xE2D1,
                _ => 0xE2E4,
            },
            Location::Init => match fvs {
                mongol::FVS1 => {
                    if wants_tooth(below) {
                        0xE2E2
                    } else {
                        0xE2E5
                    }
                }
                mongol::FVS2 => {
                    if is_ou_vowel(below) {
                        0xE2E6
                    } else {
                        0xE2E3
                    }
                }
                mongol::FVS4 => {
                    if is_ou_vowel(below) {
                        0xE2D5
                    } else {
                        0xE2D1
                    }
                }
                _ => {
                    if selects_feminine_qg(below) {
                        if is_ou_vowel(below) {
                            0xE2E6
                        } else {
                            0xE2E3
                        }
                    } else if wants_tooth(below) {
                        0xE2E1
                    } else {
                        0xE2E4
                    }
                }
            },
            Location::Medi => match fvs {
                mongol::FVS1 => {
                    if wants_tooth(below) {
                        0xE2EA
                    } else {
                        0xE2EC
                    }
                }
                mongol::FVS2 => {
                    if is_ou_vowel(below) {
                        0xE2ED
                    } else if wants_tooth(below) {
                        0xE2EB
                    } else {
                        0xE2F0
                    }
                }
                mongol::FVS3 => 0xE2EE,
                mongol::FVS4 => {
                    if is_ou_vowel(below) {
                        0xE2DE
                    } else if mongol::is_consonant(below) {
                        0xE2E0
                    } else {
                        0xE2DB
                    }
                }
                _ => {
                    if selects_feminine_qg(below) {
                        if is_ou_vowel(below) {
                            0xE2ED
                        } else {
                            0xE2EB
                        }
                    } else if is_ou_vowel(below) {
                        0xE2EC
                    } else {
                        0xE2EE
                    }
                }
            },
            Location::Fina => match fvs {
                mongol::FVS1 => 0xE2E7,
                mongol::FVS3 => 0xE2E9,
                _ => 0xE2E8,
            },
        };
    }

    match location {
        Location::Isol => match fvs {
            mongol::FVS1 => 0xE2D3,
            mongol::FVS2 => 0xE2D0,
            mongol::FVS4 => 0xE2D1,
            _ => 0xE2D2,
        },
        Location::Init => match fvs {
            mongol::FVS1 => {
                if wants_tooth(below) {
                    0xE2CF
                } else {
                    0xE2D3
                }
            }
            mongol::FVS2 => {
                if is_ou_vowel(below) {
                    0xE2D4
                } else {
                    0xE2D0
                }
            }
            mongol::FVS4 => {
                if is_ou_vowel(below) {
                    0xE2D5
                } else {
                    0xE2D1
                }
            }
            _ => {
                if selects_feminine_qg(below) {
                    if is_ou_vowel(below) {
                        0xE2D4
                    } else {
                        0xE2D0
                    }
                } else if wants_tooth(below) {
                    0xE2CE
                } else {
                    0xE2D2
                }
            }
        },
        Location::Medi => match fvs {
            mongol::FVS1 => 0xE2D9,
            mongol::FVS2 => {
                if is_ou_vowel(below) {
                    0xE2DD
                } else if mongol::is_consonant(below) {
                    0xE2DF
                } else {
                    0xE2DA
                }
            }
            mongol::FVS4 => {
                if is_ou_vowel(below) {
                    0xE2DE
                } else if mongol::is_consonant(below) {
                    0xE2E0
                } else {
                    0xE2DB
                }
            }
            _ => {
                if below == mongol::MVS {
                    0xE2D6
                } else if selects_feminine_qg(below) {
                    if is_ou_vowel(below) {
                        0xE2DD
                    } else {
                        0xE2DA
                    }
                } else if wants_tooth(below) {
                    0xE2D8
                } else {
                    0xE2DC
                }
            }
        },
        Location::Fina => {
            if fvs == mongol::FVS1 {
                0xE2D7
            } else {
                0xE2D6
            }
        }
    }
}

fn map_mala(
    location: Location,
    above: u32,
    below: u32,
    isol: u32,
    init_tooth: u32,
    init_stem: u32,
    medi_tooth: u32,
    medi_stem: u32,
    medi_bp: u32,
    fina: u32,
) -> u32 {
    match location {
        Location::Isol => isol,
        Location::Init => {
            if wants_tooth(below) {
                init_tooth
            } else {
                init_stem
            }
        }
        Location::Medi => {
            if below == mongol::MVS {
                fina
            } else if is_round_letter(above) || above == mongol::ANG {
                medi_bp
            } else if wants_tooth(below) {
                medi_tooth
            } else {
                medi_stem
            }
        }
        Location::Fina => fina,
    }
}

fn map_simple_consonant(
    location: Location,
    below: u32,
    fvs: u32,
    isol: u32,
    init_tooth: u32,
    init_stem: u32,
    medi_tooth: u32,
    medi_stem: u32,
    fina: u32,
    fina_fvs1: u32,
) -> u32 {
    match location {
        Location::Isol => isol,
        Location::Init => {
            if wants_tooth(below) {
                init_tooth
            } else {
                init_stem
            }
        }
        Location::Medi => {
            if below == mongol::MVS {
                if fvs == mongol::FVS1 {
                    fina_fvs1
                } else {
                    fina
                }
            } else if wants_tooth(below) {
                medi_tooth
            } else {
                medi_stem
            }
        }
        Location::Fina => {
            if fvs == mongol::FVS1 {
                fina_fvs1
            } else {
                fina
            }
        }
    }
}

fn map_four(location: Location, isol: u32, init: u32, medi: u32, fina: u32) -> u32 {
    match location {
        Location::Isol => isol,
        Location::Init => init,
        Location::Medi => medi,
        Location::Fina => fina,
    }
}

fn context_calls_for_double_tooth_i(word: &[u32], index: usize, above: u32, below: u32) -> bool {
    if below == mongol::I {
        return false;
    }
    matches!(above, mongol::A | mongol::E | mongol::O | mongol::U)
        || (matches!(above, mongol::OE | mongol::UE)
            && index > 0
            && !mongol::needs_long_tooth_u(word, index - 1))
}

/// Drop the glide ᠶ of a vowel+ᠶᠢ cluster. Menksoft writes that whole
/// cluster as the double-tooth i (E281) with no separate yod glyph — real
/// Menksoft text layers never put a ya-code before an i-code — so the ᠶ
/// must not emit its own letterform; the following ᠢ then sees the vowel
/// as ``above`` and takes E281 via the double-tooth rule. Word-initial and
/// FVS-marked ᠶ (᠊ᠶ᠋ᠢᠨ etc.) keep their explicit glyphs.
fn strip_glide_ya(word: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(word.len());
    for (index, &cp) in word.iter().enumerate() {
        if cp == mongol::YA
            && !word.get(index + 1).copied().is_some_and(mongol::is_fvs)
            && matches!(
                previous_base(word, index),
                mongol::A | mongol::E | mongol::O | mongol::U | mongol::OE | mongol::UE
            )
            && next_base(word, index) == mongol::I
        {
            continue;
        }
        out.push(cp);
    }
    out
}

fn previous_base(word: &[u32], index: usize) -> u32 {
    word[..index]
        .iter()
        .rev()
        .copied()
        .find(|&cp| !mongol::is_fvs(cp))
        .unwrap_or(0)
}

fn next_base(word: &[u32], index: usize) -> u32 {
    word[index + 1..]
        .iter()
        .copied()
        .find(|&cp| !mongol::is_fvs(cp))
        .unwrap_or(0)
}

fn choose_tooth_round_stem(below: u32, tooth: u32, round: u32, stem: u32) -> u32 {
    if wants_tooth(below) {
        tooth
    } else if is_round_letter_including_qg(below) {
        round
    } else {
        stem
    }
}

fn wants_tooth(codepoint: u32) -> bool {
    matches!(
        codepoint,
        mongol::A
            | mongol::E
            | mongol::I
            | mongol::OE
            | mongol::UE
            | mongol::NA
            | mongol::MA
            | mongol::LA
            | mongol::SA
            | mongol::SHA
            | mongol::TA
            | mongol::DA
            | mongol::YA
            | mongol::RA
            | mongol::WA
            | mongol::TSA
            | mongol::ZA
            | mongol::HAA
            | mongol::ZRA
            | mongol::LHA
            | mongol::ZHI
            | mongol::CHI
    )
}

fn is_ou_vowel(codepoint: u32) -> bool {
    matches!(codepoint, mongol::O | mongol::U | mongol::OE | mongol::UE)
}

/// ᠬ/ᠭ take their feminine letterforms — the glyphs FVS2 forces by hand —
/// before the feminine vowels ᠡ/ᠥ/ᠦ and before the neutral ᠢ: ki/gi is
/// written with the feminine form regardless of the word's vowel class
/// (cf. ᠬᠢᠲᠠᠳ, and the ᠊ᠳᠠᠬᠢ fixed sequence which uses E2DA).
fn selects_feminine_qg(codepoint: u32) -> bool {
    matches!(codepoint, mongol::E | mongol::OE | mongol::UE | mongol::I)
}

fn is_round_letter_including_qg(codepoint: u32) -> bool {
    is_round_letter(codepoint) || matches!(codepoint, mongol::QA | mongol::GA)
}

fn is_round_letter(codepoint: u32) -> bool {
    matches!(
        codepoint,
        mongol::BA | mongol::PA | mongol::FA | mongol::KA | mongol::KHA
    )
}

fn is_mongolian_segment_character(codepoint: u32) -> bool {
    is_mongolian_word_character(codepoint)
        || is_todo_sibe_manchu(codepoint)
        || codepoint == mongol::NNBS
        || codepoint == mongol::ZWJ
}

fn is_mongolian_word_character(codepoint: u32) -> bool {
    mongol::is_control(codepoint)
        || mongol::is_letter(codepoint)
        || matches!(codepoint, mongol::BIRGA | mongol::NIRUGU)
}

fn is_todo_sibe_manchu(codepoint: u32) -> bool {
    (0x1843..=0x18AA).contains(&codepoint)
}

fn map_punctuation(codepoint: u32) -> Option<u32> {
    Some(match codepoint {
        mongol::VERTICAL_COMMA => 0xE25D,
        mongol::VERTICAL_COLON => 0xE238,
        mongol::FULLWIDTH_SEMICOLON => 0xE252,
        mongol::FULLWIDTH_EXCLAMATION => 0xE250,
        mongol::FULLWIDTH_QUESTION => 0xE251,
        mongol::VERTICAL_EM_DASH => 0xE261,
        mongol::VERTICAL_EN_DASH => 0xE260,
        mongol::FULLWIDTH_LEFT_PARENTHESIS => 0xE253,
        mongol::FULLWIDTH_RIGHT_PARENTHESIS => 0xE254,
        mongol::VERTICAL_LEFT_TORTOISE_SHELL_BRACKET => 0xE257,
        mongol::VERTICAL_RIGHT_TORTOISE_SHELL_BRACKET => 0xE258,
        mongol::LEFT_DOUBLE_ANGLE_BRACKET => 0xE259,
        mongol::RIGHT_DOUBLE_ANGLE_BRACKET => 0xE25A,
        mongol::LEFT_ANGLE_BRACKET => 0xE255,
        mongol::RIGHT_ANGLE_BRACKET => 0xE256,
        mongol::VERTICAL_LEFT_WHITE_CORNER_BRACKET => 0xE25B,
        mongol::VERTICAL_RIGHT_WHITE_CORNER_BRACKET => 0xE25C,
        mongol::MIDDLE_DOT => 0xE243,
        mongol::REFERENCE_MARK => 0xE25F,
        mongol::QUESTION_EXCLAMATION => 0xE24E,
        mongol::EXCLAMATION_QUESTION => 0xE24F,
        mongol::ELLIPSIS => 0xE235,
        mongol::COMMA => 0xE236,
        mongol::FULL_STOP => 0xE237,
        mongol::COLON => 0xE238,
        mongol::FOUR_DOTS => 0xE239,
        mongol::TODO_SOFT_HYPHEN => 0xE23A,
        mongol::SIBE_SYLLABLE_BOUNDARY_MARKER => 0xE23B,
        mongol::MANCHU_COMMA => 0xE23C,
        mongol::MANCHU_FULL_STOP => 0xE23D,
        mongol::DIGIT_ZERO => 0xE244,
        mongol::DIGIT_ONE => 0xE245,
        mongol::DIGIT_TWO => 0xE246,
        mongol::DIGIT_THREE => 0xE247,
        mongol::DIGIT_FOUR => 0xE248,
        mongol::DIGIT_FIVE => 0xE249,
        mongol::DIGIT_SIX => 0xE24A,
        mongol::DIGIT_SEVEN => 0xE24B,
        mongol::DIGIT_EIGHT => 0xE24C,
        mongol::DIGIT_NINE => 0xE24D,
        mongol::PUNCTUATION_X => 0xE25E,
        _ => return None,
    })
}

fn from_codepoints(codepoints: &[u32]) -> String {
    codepoints
        .iter()
        .filter_map(|&cp| char::from_u32(cp))
        .collect()
}
