use encoding_mapping::{
    convert_menksoft_to_unicode, convert_mw_to_unicode, convert_unicode_to_menksoft,
    normalize_to_nominal_unicode,
};

#[test]
fn unicode_to_menksoft_matches_onon_samples() {
    let cases = [
        (
            "\u{182A}\u{1822}\u{1834}\u{1822}\u{182D}",
            "\u{E2C1}\u{E27F}\u{E317}\u{E27E}\u{E2E8}",
        ),
        (
            "\u{182E}\u{1823}\u{1829}\u{182D}\u{1823}\u{182F}",
            "\u{E2F2}\u{E289}\u{E2BC}\u{E2EC}\u{E289}\u{E2F9}",
        ),
        (
            "\u{182A}\u{1820}\u{182D}\u{180D}\u{1830}\u{1822}",
            "\u{E2C1}\u{E26D}\u{E2EE}\u{E301}\u{E27B}",
        ),
        (
            "\u{1828}\u{1821}\u{1837}\u{180E}\u{1821}",
            "\u{E2B1}\u{E276}\u{E325}\u{E274}",
        ),
    ];

    for (unicode, menksoft) in cases {
        assert_eq!(convert_unicode_to_menksoft(unicode), menksoft);
        assert_eq!(
            normalize_to_nominal_unicode(&convert_menksoft_to_unicode(menksoft)),
            normalize_to_nominal_unicode(unicode)
        );
    }
}

#[test]
fn qg_take_feminine_forms_before_feminine_vowels_and_i() {
    // ᠬ/ᠭ letterform selection is vowel-harmony sensitive: a following
    // ᠡ/ᠥ/ᠦ — and the neutral ᠢ, regardless of word class — selects the
    // feminine glyphs, while ᠠ/ᠣ/ᠤ keep the default (masculine) ones.
    // Picking the masculine form everywhere is a spelling error in every
    // feminine word (ᠭᠡᠷ rendered with the dotted masculine γ etc.).
    let cases = [
        // ken: initial feminine q + medial e
        ("\u{182C}\u{1821}\u{1828}", "\u{E2D0}\u{E277}\u{E2B5}"),
        // ger: initial feminine g
        ("\u{182D}\u{1821}\u{1837}", "\u{E2E3}\u{E277}\u{E325}"),
        // delekei: medial feminine q
        (
            "\u{1833}\u{1821}\u{182F}\u{1821}\u{182C}\u{1821}\u{1822}",
            "\u{E30E}\u{E276}\u{E2FA}\u{E276}\u{E2DA}\u{E277}\u{E27B}",
        ),
        // kümün: initial feminine q before round ü
        (
            "\u{182C}\u{1826}\u{182E}\u{1826}\u{1828}",
            "\u{E2D4}\u{E2AA}\u{E2F4}\u{E2AB}\u{E2B5}",
        ),
        // üge: medial feminine g before e
        ("\u{1826}\u{182D}\u{1821}", "\u{E2A2}\u{E2EB}\u{E275}"),
        // daki: masculine word, but q before neutral i is feminine —
        // matches the GB/T ᠊ᠳᠠᠬᠢ fixed sequence (E2DA E27C).
        (
            "\u{1833}\u{1820}\u{182C}\u{1822}",
            "\u{E30E}\u{E26C}\u{E2DA}\u{E27C}",
        ),
        // han: masculine forms stay untouched
        ("\u{182C}\u{1820}\u{1828}", "\u{E2CE}\u{E26C}\u{E2B5}"),
        // gal: masculine dotted γ stays untouched
        ("\u{182D}\u{1820}\u{182F}", "\u{E2E1}\u{E26C}\u{E2F9}"),
    ];

    for (unicode, menksoft) in cases {
        assert_eq!(convert_unicode_to_menksoft(unicode), menksoft);
        assert_eq!(
            normalize_to_nominal_unicode(&convert_menksoft_to_unicode(menksoft)),
            normalize_to_nominal_unicode(unicode)
        );
    }
}

#[test]
fn vowel_ya_i_glide_fuses_into_double_tooth_i() {
    // ᠰᠠᠶᠢᠨ and ᠰᠠᠢᠨ are homograph spellings: after a vowel the ᠶᠢ glide is
    // written as one double-tooth i (E281), never as a ya glyph plus an
    // extra tooth — real Menksoft text never puts a ya-code before an
    // i-code. Decoding the shared glyphs yields the bare-i spelling.
    let sayin = "\u{1830}\u{1820}\u{1836}\u{1822}\u{1828}";
    let sain = "\u{1830}\u{1820}\u{1822}\u{1828}";
    let fused = "\u{E2FD}\u{E26C}\u{E281}\u{E2B5}";
    assert_eq!(convert_unicode_to_menksoft(sayin), fused);
    assert_eq!(convert_unicode_to_menksoft(sain), fused);
    assert_eq!(
        normalize_to_nominal_unicode(&convert_menksoft_to_unicode(fused)),
        sain
    );
    // Consonantal ᠶ before a vowel keeps its own glyph (ᠪᠠᠶᠠᠨ)…
    assert_eq!(
        convert_unicode_to_menksoft("\u{182A}\u{1820}\u{1836}\u{1820}\u{1828}"),
        "\u{E2C1}\u{E26D}\u{E320}\u{E26C}\u{E2B5}"
    );
    // …and so does word-initial ᠶᠢ (ᠶᠢᠰᠦ).
    assert_eq!(
        convert_unicode_to_menksoft("\u{1836}\u{1822}\u{1830}\u{1826}"),
        "\u{E31E}\u{E27E}\u{E301}\u{E2A3}"
    );
}

#[test]
fn detached_suffixes_keep_their_separator_gap() {
    // The suffix-marking MVS sits inside the is_control range and used to be
    // dropped for every word the fixed table does not list, gluing ᠊ᠪᠠᠨ /
    // ᠊ᠲᠠᠢ / ᠊ᠲᠡᠬᠢ onto their stems. It must surface as the Menksoft gap
    // E263 with the first suffix letter in its suffix-initial form, while a
    // separated final vowel keeps carrying the gap inside its own glyph
    // (E26A/E274) with no E263.
    let cases = [
        // ᠊ᠪᠠᠨ: no dedicated suffix-initial form — gap + regular init
        (
            "\u{180E}\u{182A}\u{1820}\u{1828}",
            "\u{E263}\u{E2C1}\u{E26D}\u{E2B5}",
        ),
        // ger-teki: feminine g + gap + suffix-initial ᠲ + feminine ᠬᠢ
        (
            "\u{182D}\u{1821}\u{1837}\u{180E}\u{1832}\u{1821}\u{182C}\u{1822}",
            "\u{E2E3}\u{E277}\u{E325}\u{E263}\u{E309}\u{E276}\u{E2DA}\u{E27C}",
        ),
        // mori-tai: gap + suffix-initial ᠲ
        (
            "\u{182E}\u{1823}\u{1837}\u{1822}\u{180E}\u{1832}\u{1820}\u{1822}",
            "\u{E2F2}\u{E289}\u{E327}\u{E27B}\u{E263}\u{E309}\u{E26C}\u{E27B}",
        ),
        // gajar-a: separated final vowel — no E263, gap is in the glyph
        (
            "\u{182D}\u{1820}\u{1835}\u{1820}\u{1837}\u{180E}\u{1820}",
            "\u{E2E1}\u{E26C}\u{E31D}\u{E26C}\u{E325}\u{E26A}",
        ),
    ];

    for (unicode, menksoft) in cases {
        assert_eq!(convert_unicode_to_menksoft(unicode), menksoft);
        assert_eq!(
            normalize_to_nominal_unicode(&convert_menksoft_to_unicode(menksoft)),
            normalize_to_nominal_unicode(unicode)
        );
    }
}

#[test]
fn mw_and_menksoft_collapse_to_same_nominal_unicode() {
    let mw = "\u{182A}\u{1820}\u{182D}\u{180D}\u{1830}\u{1822}";
    let menksoft = "\u{E2C1}\u{E26D}\u{E2EE}\u{E301}\u{E27B}";
    let nominal = "\u{182A}\u{1820}\u{182D}\u{1830}\u{1822}";

    assert_eq!(normalize_to_nominal_unicode(mw), nominal);
    assert_eq!(normalize_to_nominal_unicode(menksoft), nominal);
}

#[test]
fn mw_pipeline_unifies_nnbs_and_removes_zero_width_noise() {
    let noisy_mw = "\u{FEFF}\u{1828}\u{1821}\u{1837}\u{202F}\u{200D}\u{1821}\u{200B}";

    assert_eq!(
        convert_mw_to_unicode(noisy_mw),
        "\u{1828}\u{1821}\u{1837}\u{180E}\u{1821}"
    );
    assert_eq!(
        normalize_to_nominal_unicode(noisy_mw),
        "\u{1828}\u{1821}\u{1837}\u{180E}\u{1821}"
    );
}
