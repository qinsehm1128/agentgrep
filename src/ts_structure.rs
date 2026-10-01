//! AST-based structure extraction (feature `treesitter`).
//!
//! The line-based extractors in `structure.rs` cover Rust, TS/JS, Python and
//! Markdown. Every other language used to produce no structure items, so
//! smart mode found no regions and returned nothing for Go, Java, C and so
//! on. This module fills that gap with tree-sitter, for languages the
//! line-based path does not handle. The definition node kinds per language
//! mirror qin-code's `code_index` classifiers.

#![cfg(feature = "treesitter")]

use crate::structure::StructureItem;
use tree_sitter::{Language, Node, Parser};

struct Spec {
    language: fn() -> Language,
    /// `(node kind, item kind)` pairs that define a named symbol.
    definitions: &'static [(&'static str, &'static str)],
}

const GO: Spec = Spec {
    language: || tree_sitter_go::LANGUAGE.into(),
    definitions: &[
        ("function_declaration", "function"),
        ("method_declaration", "function"),
        ("type_spec", "type"),
    ],
};

const JAVA: Spec = Spec {
    language: || tree_sitter_java::LANGUAGE.into(),
    definitions: &[
        ("class_declaration", "class"),
        ("interface_declaration", "trait"),
        ("enum_declaration", "enum"),
        ("record_declaration", "class"),
        ("method_declaration", "function"),
        ("constructor_declaration", "function"),
    ],
};

const C: Spec = Spec {
    language: || tree_sitter_c::LANGUAGE.into(),
    definitions: &[
        ("function_definition", "function"),
        ("struct_specifier", "struct"),
        ("union_specifier", "struct"),
        ("enum_specifier", "enum"),
        ("type_definition", "type"),
    ],
};

const CPP: Spec = Spec {
    language: || tree_sitter_cpp::LANGUAGE.into(),
    definitions: &[
        ("function_definition", "function"),
        ("class_specifier", "class"),
        ("struct_specifier", "struct"),
        ("union_specifier", "struct"),
        ("enum_specifier", "enum"),
        ("type_definition", "type"),
        ("alias_declaration", "type"),
        ("namespace_definition", "module"),
    ],
};

const CSHARP: Spec = Spec {
    language: || tree_sitter_c_sharp::LANGUAGE.into(),
    definitions: &[
        ("class_declaration", "class"),
        ("struct_declaration", "struct"),
        ("interface_declaration", "trait"),
        ("enum_declaration", "enum"),
        ("record_declaration", "class"),
        ("method_declaration", "function"),
        ("constructor_declaration", "function"),
    ],
};

const RUBY: Spec = Spec {
    language: || tree_sitter_ruby::LANGUAGE.into(),
    definitions: &[
        ("method", "function"),
        ("singleton_method", "function"),
        ("class", "class"),
        ("module", "module"),
    ],
};

const PHP: Spec = Spec {
    language: || tree_sitter_php::LANGUAGE_PHP.into(),
    definitions: &[
        ("class_declaration", "class"),
        ("interface_declaration", "trait"),
        ("trait_declaration", "trait"),
        ("enum_declaration", "enum"),
        ("function_definition", "function"),
        ("method_declaration", "function"),
    ],
};

const KOTLIN: Spec = Spec {
    language: || tree_sitter_kotlin_ng::LANGUAGE.into(),
    definitions: &[
        ("class_declaration", "class"),
        ("object_declaration", "class"),
        ("function_declaration", "function"),
        ("type_alias", "type"),
    ],
};

const SWIFT: Spec = Spec {
    language: || tree_sitter_swift::LANGUAGE.into(),
    definitions: &[
        ("class_declaration", "class"),
        ("protocol_declaration", "trait"),
        ("function_declaration", "function"),
        ("protocol_function_declaration", "function"),
        ("typealias_declaration", "type"),
    ],
};

/// Language name for a file extension, if this module handles it.
pub fn language_for_extension(ext: &str) -> Option<&'static str> {
    Some(match ext {
        "go" => "go",
        "java" => "java",
        "c" | "h" => "c",
        "cc" | "cpp" | "cxx" | "hpp" | "hh" | "hxx" => "cpp",
        "cs" => "csharp",
        "rb" => "ruby",
        "php" => "php",
        "kt" | "kts" => "kotlin",
        "swift" => "swift",
        _ => return None,
    })
}

fn spec(language: &str) -> Option<&'static Spec> {
    Some(match language {
        "go" => &GO,
        "java" => &JAVA,
        "c" => &C,
        "cpp" => &CPP,
        "csharp" => &CSHARP,
        "ruby" => &RUBY,
        "php" => &PHP,
        "kotlin" => &KOTLIN,
        "swift" => &SWIFT,
        _ => return None,
    })
}

/// Files larger than this are left to the line-based path.
const MAX_PARSE_BYTES: usize = 2 * 1024 * 1024;

/// Extract definition items with exact line ranges. `None` if the language
/// is not handled here or parsing fails.
pub fn extract(language: &str, text: &str) -> Option<Vec<StructureItem>> {
    let spec = spec(language)?;
    if text.len() > MAX_PARSE_BYTES {
        return None;
    }
    let mut parser = Parser::new();
    parser.set_language(&(spec.language)()).ok()?;
    let tree = parser.parse(text, None)?;
    let mut items = Vec::new();
    collect(tree.root_node(), text.as_bytes(), spec, &mut items);
    items.sort_by(|a, b| {
        a.start_line
            .cmp(&b.start_line)
            .then(b.end_line.cmp(&a.end_line))
    });
    Some(items)
}

fn collect(node: Node<'_>, src: &[u8], spec: &Spec, out: &mut Vec<StructureItem>) {
    let kind = node.kind();
    if let Some((_, item_kind)) = spec.definitions.iter().find(|(k, _)| *k == kind)
        && let Some(label) = definition_name(node, src)
    {
        let start_line = node.start_position().row + 1;
        let end_line = node.end_position().row + 1;
        out.push(StructureItem {
            kind: (*item_kind).to_string(),
            label,
            start_line,
            end_line,
            line_count: end_line - start_line + 1,
        });
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect(child, src, spec, out);
    }
}

/// The defined name: the `name` field when the grammar has one, otherwise
/// the identifier inside a C-style `declarator` chain.
fn definition_name(node: Node<'_>, src: &[u8]) -> Option<String> {
    if let Some(name) = node.child_by_field_name("name") {
        return text_of(name, src);
    }
    let mut current = node.child_by_field_name("declarator")?;
    loop {
        match current.kind() {
            "identifier"
            | "field_identifier"
            | "type_identifier"
            | "qualified_identifier"
            | "destructor_name"
            | "operator_name" => return text_of(current, src),
            _ => current = current.child_by_field_name("declarator")?,
        }
    }
}

fn text_of(node: Node<'_>, src: &[u8]) -> Option<String> {
    let text = node.utf8_text(src).ok()?.trim();
    (!text.is_empty()).then(|| text.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labels(language: &str, text: &str) -> Vec<(String, usize, usize)> {
        extract(language, text)
            .unwrap()
            .into_iter()
            .map(|i| (i.label, i.start_line, i.end_line))
            .collect()
    }

    #[test]
    fn go_functions_methods_and_types() {
        let text = "package p\n\ntype Store struct {\n\tk int\n}\n\nfunc (s *Store) Refresh() error {\n\treturn nil\n}\n\nfunc helper() {}\n";
        assert_eq!(
            labels("go", text),
            vec![
                ("Store".into(), 3, 5),
                ("Refresh".into(), 7, 9),
                ("helper".into(), 11, 11)
            ]
        );
    }

    #[test]
    fn c_function_name_from_declarator_chain() {
        let text = "static int *parse_header(const char *buf) {\n    return 0;\n}\n";
        assert_eq!(labels("c", text), vec![("parse_header".into(), 1, 3)]);
    }

    #[test]
    fn java_class_and_method_nest() {
        let text = "class Billing {\n  int lateFee() {\n    return 1;\n  }\n}\n";
        assert_eq!(
            labels("java", text),
            vec![("Billing".into(), 1, 5), ("lateFee".into(), 2, 4)]
        );
    }

    #[test]
    fn every_language_parses() {
        for (lang, text) in [
            ("go", "package p\nfunc A() {}\n"),
            ("java", "class A { void b() {} }\n"),
            ("c", "int a(void) { return 0; }\n"),
            ("cpp", "namespace n { int a() { return 0; } }\n"),
            ("csharp", "class A { void B() {} }\n"),
            ("ruby", "class A\n  def b; end\nend\n"),
            ("php", "<?php\nfunction a() {}\n"),
            ("kotlin", "fun a() {}\n"),
            ("swift", "func a() {}\n"),
        ] {
            let items = extract(lang, text).unwrap_or_default();
            assert!(!items.is_empty(), "{lang} produced no items");
        }
    }

    #[test]
    fn unknown_language_is_none() {
        assert!(extract("rust", "fn a() {}").is_none());
        assert!(language_for_extension("rs").is_none());
    }
}
