// usage_macro_analyzer.cpp - Accurate resolution of uses at macro expansion time + post-AST deferred resolution
#include "clang/AST/ASTConsumer.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Basic/FileEntry.h"
#include "llvm/ADT/SmallVector.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Expr.h"
#include <iostream>
#include <map>
#include <set>
#include <vector>
#include <mutex>

#if __cplusplus >= 201703L && __has_include(<filesystem>)
    #include <filesystem>
    namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
    #include <experimental/filesystem>
    namespace fs = std::experimental::filesystem;
#else
    #error "No filesystem support"
#endif

using namespace clang;
using namespace clang::tooling;

static std::string g_compileDir;

static const std::vector<std::string> DEFAULT_INCLUDE_PATHS = {
    "-resource-dir=/usr/lib/llvm-19/lib/clang/19",
    "-w",
    "-Wno-incompatible-function-pointer-types", 
    "-Wno-incompatible-pointer-types",
    // "-isystem/usr/include/c++/11",
    // "-isystem/usr/include/x86_64-linux-gnu/c++/11",
    // "-isystem/usr/include/c++/11/backward",
    "-isystem/usr/include/x86_64-linux-gnu",
    "-isystem/usr/include",
    "-fno-strict-aliasing",
};


struct UseInfo {
    std::string kind;
    std::string name;
    std::string usage_location;
    std::string definition;
    int start_line = 0;
    int end_line = 0;
    std::string file_path;
    bool pending_ast_resolve = false;
    unsigned expansion_raw_loc = 0;

    bool operator==(const UseInfo &o) const {
        return name == o.name && definition == o.definition;
    }
};


struct UsesPattern {
    std::vector<UseInfo> uses;

    bool operator<(const UsesPattern &o) const {
        if (uses.size() != o.uses.size()) return uses.size() < o.uses.size();
        for (size_t i = 0; i < uses.size(); ++i) {
            if (uses[i].name != o.uses[i].name) return uses[i].name < o.uses[i].name;
            if (uses[i].definition != o.uses[i].definition) return uses[i].definition < o.uses[i].definition;
        }
        return false;
    }
    bool operator==(const UsesPattern &o) const {
        if (uses.size() != o.uses.size()) return false;
        for (size_t i = 0; i < uses.size(); ++i) {
            if (!(uses[i] == o.uses[i])) return false;
        }
        return true;
    }
};

struct MacroKey {
    std::string name;
    std::string definition_location;
    UsesPattern uses_pattern;

    bool operator<(const MacroKey &o) const {
        if (name != o.name) return name < o.name;
        if (definition_location != o.definition_location) return definition_location < o.definition_location;
        return uses_pattern < o.uses_pattern;
    }
};

struct MacroEntry {
    std::string name;
    std::string definition_location;
    int start_line = 0;
    int end_line = 0;
    std::string file_path;
    std::set<std::string> appearances;
    std::string undef_location;
    std::string expanded_value;
    std::string kind;
    std::vector<std::string> parameters;
    bool isConst = false;
    std::vector<UseInfo> uses;
};


struct MacroDefInfo {
    std::string name;
    std::string definition_location;
    int start_line = 0;
    int end_line = 0;
    std::string file_path;
    std::string expanded_value;
    std::string kind;
    std::vector<std::string> parameters;
    std::string undef_location;
    const MacroInfo *MI = nullptr;
    bool isConst = false; 
};

struct PendingASTResolve {
    std::string macro_name;
    std::string macro_def_location;
    std::string token_name;
    std::string token_usage_location;
    unsigned expansion_raw_loc;
    std::string appearance_location;
};

static std::map<MacroKey, MacroEntry> globalMacros;
static std::map<std::string, MacroDefInfo> macroDefByLocation;
static std::map<std::string, std::string> currentActiveMacro;
static std::vector<PendingASTResolve> pendingASTResolves;
static std::mutex macrosMutex;


class MacroAnalyzer {
public:
    SourceManager *SM;
    
    explicit MacroAnalyzer(SourceManager *SM) : SM(SM) {}
    
    std::string getAbsolutePath(StringRef filename) {

        // added: Prepend compile_dir to virtual paths such as <built-in> and <command line>
        std::string name = filename.str();
        if (!name.empty() && name[0] == '<') {
            if (!g_compileDir.empty()) {
                return g_compileDir + "/" + name;
            }
            return name;
        }
        // ended

        std::filesystem::path p(filename.str());
        std::error_code ec;
        if (!p.is_absolute()) {
            p = std::filesystem::absolute(p, ec);
        }
        return p.lexically_normal().string();
    }
    
    std::string getOriginalLocationString(SourceLocation Loc) {
        if (Loc.isInvalid()) return "unknown";
        
        SourceLocation OriginalLoc = Loc;
        
        if (Loc.isMacroID()) {
            SourceLocation CustomOriginal = SM->getOriginalLoc(Loc);
            if (CustomOriginal.isFileID()) {
                OriginalLoc = CustomOriginal;
            } else {
                OriginalLoc = SM->getSpellingLoc(Loc);
            }
        }

        // added: Retrieve the physical location via SpellingLoc, and fall back to PresumedLoc when FileEntryRef retrieval fails
        SourceLocation SpellingLoc = SM->getSpellingLoc(OriginalLoc);
        FileID FID = SM->getFileID(SpellingLoc);
        OptionalFileEntryRef FER = SM->getFileEntryRefForID(FID);
        
        if (FER) {
            std::string filePath = getAbsolutePath(FER->getName());
            unsigned line = SM->getLineNumber(FID, SM->getFileOffset(SpellingLoc));
            unsigned column = SM->getColumnNumber(FID, SM->getFileOffset(SpellingLoc));
            
            std::string Result;
            llvm::raw_string_ostream OS(Result);
            OS << filePath << ":" << line << ":" << column;
            return OS.str();
        }
        
        PresumedLoc PLoc = SM->getPresumedLoc(OriginalLoc);
        if (PLoc.isValid()) {
            std::string Result;
            llvm::raw_string_ostream OS(Result);
            OS << PLoc.getFilename() << ":" << PLoc.getLine() << ":" << PLoc.getColumn();
            return OS.str();
        }
        // ended
        
        return "unknown";
    }
    
    std::string getFilePath(SourceLocation Loc) {
        if (Loc.isInvalid()) return "";
        
        SourceLocation OriginalLoc = Loc;
        if (Loc.isMacroID()) {
            OriginalLoc = SM->getSpellingLoc(Loc);
        }
        
        FileID FID = SM->getFileID(OriginalLoc);
        OptionalFileEntryRef FER = SM->getFileEntryRefForID(FID);

        // added: Fall back to PresumedLoc when FileEntryRef retrieval fails
        if (FER) {
            return getAbsolutePath(FER->getName());
        }
        
        PresumedLoc PLoc = SM->getPresumedLoc(OriginalLoc);
        if (PLoc.isValid()) {
            return PLoc.getFilename();
        }
        // ended

        return "";
    }
    
    int getLineNumber(SourceLocation Loc) {
        if (Loc.isInvalid()) return 0;
        
        SourceLocation OriginalLoc = Loc;
        if (Loc.isMacroID()) {
            OriginalLoc = SM->getSpellingLoc(Loc);
        }
        
        FileID FID = SM->getFileID(OriginalLoc);

        // added: Fall back to PresumedLoc when FileEntryRef retrieval fails
        OptionalFileEntryRef FER = SM->getFileEntryRefForID(FID);
        if (FER) {
            return SM->getLineNumber(FID, SM->getFileOffset(OriginalLoc));
        }
        
        PresumedLoc PLoc = SM->getPresumedLoc(OriginalLoc);
        if (PLoc.isValid()) {
            return PLoc.getLine();
        }
        // ended
        
        return 0;
    }
    
    bool isSystemLocation(SourceLocation Loc) {
        if (Loc.isInvalid()) return true;
        
        SourceLocation OriginalLoc = Loc;
        if (Loc.isMacroID()) {
            OriginalLoc = SM->getOriginalLoc(Loc);
            if (OriginalLoc.isMacroID()) {
                OriginalLoc = SM->getSpellingLoc(Loc);
            }
        }
        return SM->isInSystemHeader(OriginalLoc);
    }
};

class MacroCallback : public PPCallbacks {
private:
    MacroAnalyzer &Analyzer;
    Preprocessor &PP;

public:
    MacroCallback(MacroAnalyzer &Analyzer, Preprocessor &PP) 
        : Analyzer(Analyzer), PP(PP) {}

    
    void MacroDefined(const Token &MacroNameTok, const MacroDirective *MD) override {
        SourceLocation MacroLoc = MacroNameTok.getLocation();
        
        std::string macroName = MacroNameTok.getIdentifierInfo()->getName().str();
        std::string defLocation = Analyzer.getOriginalLocationString(MacroLoc);
        
        const clang::MacroInfo *MI = MD->getMacroInfo();

        MacroDefInfo defInfo;
        defInfo.name = macroName;
        defInfo.definition_location = defLocation;
        defInfo.MI = MI;
        
        if (MI) {
            SourceLocation DefStart = MI->getDefinitionLoc();
            SourceLocation DefEnd = MI->getDefinitionEndLoc();
            
            if (DefStart.isValid()) {
                defInfo.file_path = Analyzer.getFilePath(DefStart);
                defInfo.start_line = Analyzer.getLineNumber(DefStart);
                
                if (DefEnd.isValid()) {
                    defInfo.end_line = Analyzer.getLineNumber(DefEnd);
                } else {
                    defInfo.end_line = defInfo.start_line;
                }
            }

            std::string expandedValue;
            for (const Token &Tok : MI->tokens()) {
                if (!expandedValue.empty()) {
                    expandedValue += " ";
                }
                
                if (Tok.is(tok::identifier)) {
                    expandedValue += Tok.getIdentifierInfo()->getName().str();
                } else if (Tok.is(tok::numeric_constant) || Tok.is(tok::string_literal)) {
                    expandedValue += std::string(Tok.getLiteralData(), Tok.getLength());
                } else {
                    expandedValue += PP.getSpelling(Tok);
                }
            }
            defInfo.expanded_value = expandedValue;

            if (MI->isFunctionLike()) {
                defInfo.kind = "macro_function";
                for (const IdentifierInfo *Param : MI->params()) {
                    if (Param) {
                        defInfo.parameters.push_back(Param->getName().str());
                    }
                }
            } else if (MI->getNumTokens() == 0) {
                defInfo.kind = "macro_flag";
            } else {
                defInfo.kind = "macro";
            }

            // added
            // Token-sequence pattern matching for preliminary is_const determination (bindgen-compatible)
            if (!MI->isFunctionLike() && MI->getNumTokens() > 0) {
                bool allNumericOrOp = true;
                bool hasNumeric = false;
                bool hasFloat = false;
                bool isSingleString = (MI->getNumTokens() == 1 && 
                                       MI->tokens()[0].is(tok::string_literal));

                for (const Token &Tok : MI->tokens()) {
                    if (Tok.is(tok::numeric_constant)) {
                        hasNumeric = true;
                        // Detect floating-point literals
                        std::string spelling(Tok.getLiteralData(), Tok.getLength());
                        if (spelling.find('.') != std::string::npos ||
                            spelling.find('e') != std::string::npos ||
                            spelling.find('E') != std::string::npos) {
                            hasFloat = true;
                        }
                    } else if (Tok.isOneOf(tok::plus, tok::minus, tok::star, tok::slash,
                                           tok::percent, tok::amp, tok::pipe, tok::caret,
                                           tok::tilde, tok::lessless, tok::greatergreater,
                                           tok::l_paren, tok::r_paren, tok::exclaim)) {
                        // Allow operators and parentheses
                    } else if (Tok.is(tok::string_literal)) {
                        // Allow if standalone (already handled separately by isSingleString)
                    } else {
                        allNumericOrOp = false;
                    }
                }

                if (isSingleString) {
                    defInfo.isConst = true;
                } else if (hasNumeric && allNumericOrOp) {
                    defInfo.isConst = true;
                }
            }
            // ended
        }
        
        std::lock_guard<std::mutex> lock(macrosMutex);
        
        llvm::errs() << "[MacroDefined] " << macroName 
                        << " at " << defLocation << "\n";
        
        macroDefByLocation[defLocation] = defInfo;
        currentActiveMacro[macroName] = defLocation;
    }

    void MacroExpands(const Token &MacroNameTok, const MacroDefinition &MD,
                     SourceRange Range, const MacroArgs *Args) override {
        const MacroInfo *MI = MD.getMacroInfo();
        if (!MI) return;

        SourceLocation MacroLoc = MacroNameTok.getLocation();
        std::string macroName = MacroNameTok.getIdentifierInfo()->getName().str();
        std::string useLocation = Analyzer.getOriginalLocationString(MacroLoc);
        unsigned expansionRawLoc = MacroLoc.getRawEncoding();

        std::lock_guard<std::mutex> lock(macrosMutex);

        auto activeIt = currentActiveMacro.find(macroName);
        if (activeIt == currentActiveMacro.end()) {
            // External macro (preserve existing logic)
            // Register with an empty uses pattern
            UsesPattern emptyPattern;
            MacroKey extKey{macroName, "external", emptyPattern};
            if (globalMacros.find(extKey) == globalMacros.end()) {
                MacroEntry extEntry;
                extEntry.name = macroName;
                extEntry.definition_location = "external";
                extEntry.kind = MI->isFunctionLike() ? "macro_function" : "macro";
                globalMacros[extKey] = extEntry;
            }
            globalMacros[extKey].appearances.insert(useLocation);
            llvm::errs() << "[MacroExpands] " << macroName 
                         << " at " << useLocation << " (external)\n";
            return;
        }

        std::string defLocation = activeIt->second;
        auto defIt = macroDefByLocation.find(defLocation);
        if (defIt == macroDefByLocation.end()) return;
        const MacroDefInfo &defInfo = defIt->second;

        // ---- Scan body tokens to build uses ----
        std::vector<UseInfo> uses;

        std::set<std::string> paramNames;
        if (MI->isFunctionLike()) {
            for (const IdentifierInfo *Param : MI->params())
                if (Param) paramNames.insert(Param->getName().str());
        }

        for (const Token &Tok : MI->tokens()) {
            if (!Tok.is(tok::identifier)) continue;
            const IdentifierInfo *TokII = Tok.getIdentifierInfo();
            if (!TokII) continue;

            std::string tokenName = TokII->getName().str();
            if (paramNames.count(tokenName)) continue;

            UseInfo ui;
            ui.name = tokenName;
            ui.usage_location = Analyzer.getOriginalLocationString(Tok.getLocation());

            // Attempt to resolve as a macro
            IdentifierInfo *RefII = PP.getIdentifierInfo(tokenName);
            if (RefII && RefII->hasMacroDefinition()) {
                MacroDefinition RefMD = PP.getMacroDefinition(RefII);
                if (RefMD && RefMD.getMacroInfo()) {
                    const MacroInfo *RefMI = RefMD.getMacroInfo();
                    SourceLocation RefDefLoc = RefMI->getDefinitionLoc();
                    std::string refDef = Analyzer.getOriginalLocationString(RefDefLoc);

                    // Definition points to itself → defer to AST resolution
                    if (refDef == defLocation) {
                        ui.kind = "pending";
                        ui.definition = "pending";
                        ui.pending_ast_resolve = true;
                        ui.expansion_raw_loc = expansionRawLoc;

                        pendingASTResolves.push_back({
                            macroName, defLocation, tokenName,
                            ui.usage_location, expansionRawLoc, useLocation
                        });
                    } else {
                        ui.kind = RefMI->isFunctionLike() ? "macro_function" : "macro";
                        ui.definition = refDef;
                        ui.file_path = Analyzer.getFilePath(RefDefLoc);
                        PresumedLoc sl = Analyzer.SM->getPresumedLoc(RefDefLoc);
                        PresumedLoc el = Analyzer.SM->getPresumedLoc(RefMI->getDefinitionEndLoc());
                        ui.start_line = sl.isValid() ? sl.getLine() : 0;
                        ui.end_line = el.isValid() ? el.getLine() : 0;
                    }

                    uses.push_back(ui);
                    continue;
                }
            }

            // Not a macro → defer to AST resolution
            ui.kind = "pending";
            ui.definition = "pending";
            ui.pending_ast_resolve = true;
            ui.expansion_raw_loc = expansionRawLoc;

            pendingASTResolves.push_back({
                macroName, defLocation, tokenName,
                ui.usage_location, expansionRawLoc, useLocation
            });

            uses.push_back(ui);
        }

        // Build the key from the uses pattern
        UsesPattern pattern;
        pattern.uses = uses;

        MacroKey key{macroName, defLocation, pattern};

        auto it = globalMacros.find(key);
        if (it != globalMacros.end()) {
            it->second.appearances.insert(useLocation);
        } else {
            MacroEntry entry;
            entry.name = defInfo.name;
            entry.definition_location = defInfo.definition_location;
            entry.start_line = defInfo.start_line;
            entry.end_line = defInfo.end_line;
            entry.file_path = defInfo.file_path;
            entry.expanded_value = defInfo.expanded_value;
            entry.kind = defInfo.kind;
            entry.parameters = defInfo.parameters;
            entry.undef_location = defInfo.undef_location;
            entry.uses = uses;
            entry.appearances.insert(useLocation);
            entry.isConst = defInfo.isConst;
            globalMacros[key] = entry;
        }

        llvm::errs() << "[MacroExpands] " << macroName 
                     << " at " << useLocation
                     << " (uses: " << uses.size() << " tokens)\n";
    }

    void MacroUndefined(const Token &MacroNameTok, const MacroDefinition &MD,
                        const MacroDirective *Undef) override {
        std::string macroName = MacroNameTok.getIdentifierInfo()->getName().str();
        std::string undefLocation = Analyzer.getOriginalLocationString(MacroNameTok.getLocation());
        
        std::lock_guard<std::mutex> lock(macrosMutex);
        
        // Record the #undef location for the currently active macro definition
        auto it = currentActiveMacro.find(macroName);
        if (it != currentActiveMacro.end()) {
            // Also record in macroDefByLocation
            auto defIt = macroDefByLocation.find(it->second);
            if (defIt != macroDefByLocation.end()) {
                defIt->second.undef_location = undefLocation;
            }
            // Also record in the corresponding entries in globalMacros
            for (auto &[key, entry] : globalMacros) {
                if (key.name == macroName && key.definition_location == it->second) {
                    entry.undef_location = undefLocation;
                }
            }
            currentActiveMacro.erase(it);
        }
        
        llvm::errs() << "[MacroUndefined] " << macroName 
                     << " at " << undefLocation << "\n";
    }
};

// ---- AST Visitor: Resolve pending token definitions via DeclRefExpr ----
class PendingResolveVisitor : public RecursiveASTVisitor<PendingResolveVisitor> {
    ASTContext &Context;
    SourceManager &SM;
    MacroAnalyzer &Analyzer;
    std::map<std::pair<unsigned, std::string>, std::pair<std::string, const NamedDecl*>> &Resolved;

public:
    PendingResolveVisitor(
        ASTContext &Ctx, MacroAnalyzer &A,
        std::map<std::pair<unsigned, std::string>, std::pair<std::string, const NamedDecl*>> &R)
        : Context(Ctx), SM(Ctx.getSourceManager()), Analyzer(A), Resolved(R) {}

    bool VisitDeclRefExpr(DeclRefExpr *DRE) {
        SourceLocation Loc = DRE->getLocation();
        if (!Loc.isMacroID()) return true;

        SourceLocation ExpLoc = SM.getExpansionLoc(Loc);
        unsigned rawLoc = ExpLoc.getRawEncoding();

        const NamedDecl *ND = DRE->getDecl();
        if (!ND) return true;

        std::string tokenName = ND->getNameAsString();
        auto key = std::make_pair(rawLoc, tokenName);

        if (Resolved.find(key) == Resolved.end()) {
            SourceLocation DeclLoc = ND->getLocation();
            std::string declLocStr = Analyzer.getOriginalLocationString(DeclLoc);
            Resolved[key] = {declLocStr, ND};
        }

        return true;
    }
};


class MacroConstEvalVisitor 
    : public RecursiveASTVisitor<MacroConstEvalVisitor> {
    
    typedef RecursiveASTVisitor<MacroConstEvalVisitor> Base;

    ASTContext &Context;
    SourceManager &SM;
    const llvm::DenseMap<unsigned, std::pair<std::string, unsigned>> &UseSites;
    std::map<unsigned, bool> &Results;

    llvm::DenseSet<unsigned> EvaluatedLocs;

public:
    MacroConstEvalVisitor(
        ASTContext &Context,
        const llvm::DenseMap<unsigned, std::pair<std::string, unsigned>> &UseSites,
        std::map<unsigned, bool> &Results)
        : Context(Context),
          SM(Context.getSourceManager()),
          UseSites(UseSites),
          Results(Results) {}

    bool TraverseStmt(Stmt *S) {
        if (!S) return true;
        
        Expr *E = dyn_cast<Expr>(S);
        if (!E)
            return Base::TraverseStmt(S);
        
        SourceLocation Loc = E->getBeginLoc();
        
        if (!Loc.isMacroID())
            return Base::TraverseStmt(S);
        
        SourceLocation ExpansionLoc = SM.getExpansionLoc(Loc);
        unsigned RawLoc = ExpansionLoc.getRawEncoding();
        
        // added 
        auto It = UseSites.find(RawLoc);
        if (It == UseSites.end())
            return Base::TraverseStmt(S);

        // Verify that the Expr end location is also within this macro expansion
        SourceLocation EndLoc = E->getEndLoc();
        bool EndInSameMacro = EndLoc.isMacroID() &&
            SM.getExpansionLoc(EndLoc).getRawEncoding() == RawLoc;
        
        if (!EndInSameMacro)
            return Base::TraverseStmt(S);
        
        const std::string &MacroName = It->second.first;
        unsigned DefRawLoc = It->second.second;
        
        if (DefRawLoc == 0) return true;

        Expr::EvalResult EvalResult;
        bool IsConst = E->EvaluateAsRValue(EvalResult, Context);
        
        // bindgen-compatible: treat string literals as const
        if (!IsConst) {
            const Expr *Inner = E->IgnoreParens();
            if (isa<StringLiteral>(Inner) || isa<ObjCStringLiteral>(Inner)) {
                IsConst = true;
            }
        }
        
        // If already evaluated: do not overwrite true with false
        if (EvaluatedLocs.count(RawLoc)) {
            if (IsConst) {
                // Promote a previously false result to true
                auto ResultIt = Results.find(DefRawLoc);
                if (ResultIt != Results.end() && !ResultIt->second) {
                    ResultIt->second = true;
                }
            }
            return Base::TraverseStmt(S);
        }
        
        EvaluatedLocs.insert(RawLoc);
        
        auto ResultIt = Results.find(DefRawLoc);
        if (ResultIt == Results.end()) {
            Results[DefRawLoc] = IsConst;
        } else if (!IsConst) {
            ResultIt->second = false;
        }
        // ended

        // //
        // if (EvaluatedLocs.count(RawLoc))
        //     return Base::TraverseStmt(S);
        
        // auto It = UseSites.find(RawLoc);
        // if (It == UseSites.end())
        //     return Base::TraverseStmt(S);

        // SourceLocation EndLoc = E->getEndLoc();
        // bool EndInSameMacro = EndLoc.isMacroID() &&
        //     SM.getExpansionLoc(EndLoc).getRawEncoding() == RawLoc;
        
        // if (!EndInSameMacro)
        //     return Base::TraverseStmt(S);
        
        // EvaluatedLocs.insert(RawLoc);
        
        // const std::string &MacroName = It->second.first;
        // unsigned DefRawLoc = It->second.second;
        
        // if (DefRawLoc == 0) return true;

        // Expr::EvalResult EvalResult;
        // bool IsConst = E->EvaluateAsRValue(EvalResult, Context);
        
        // if (!IsConst) {
        //     const Expr *Inner = E->IgnoreParens();
        //     if (isa<StringLiteral>(Inner) || isa<ObjCStringLiteral>(Inner)) {
        //         IsConst = true;
        //     }
        // }

        // auto ResultIt = Results.find(DefRawLoc);
        // if (ResultIt == Results.end()) {
        //     Results[DefRawLoc] = IsConst;
        // } else if (!IsConst) {
        //     ResultIt->second = false;
        // }
        // //
        
        return true;
    }
};


class MacroASTConsumer : public ASTConsumer {
private:
    MacroAnalyzer Analyzer;
    Preprocessor &PP;

public:
    explicit MacroASTConsumer(SourceManager *SM, Preprocessor &PP) 
        : Analyzer(SM), PP(PP) {
        PP.addPPCallbacks(std::make_unique<MacroCallback>(Analyzer, PP));
    }

    void HandleTranslationUnit(ASTContext &Context) override {
        llvm::errs() << "Translation unit processed\n";

        // ---- Phase 1: Traverse the AST to resolve pending tokens ----
        std::map<std::pair<unsigned, std::string>, std::pair<std::string, const NamedDecl*>> resolved;
        {
            PendingResolveVisitor PRV(Context, Analyzer, resolved);
            PRV.TraverseDecl(Context.getTranslationUnitDecl());
        }

        // Apply pending resolution results to globalMacros
        {
            std::lock_guard<std::mutex> lock(macrosMutex);
            for (auto &[key, entry] : globalMacros) {
                for (auto &ui : entry.uses) {
                    if (!ui.pending_ast_resolve) continue;

                    auto rkey = std::make_pair(ui.expansion_raw_loc, ui.name);
                    auto rit = resolved.find(rkey);
                    if (rit != resolved.end()) {
                        const std::string &declLocStr = rit->second.first;
                        const NamedDecl *ND = rit->second.second;

                        ui.definition = declLocStr;
                        ui.pending_ast_resolve = false;

                        if (isa<FunctionDecl>(ND)) ui.kind = "function";
                        else if (isa<VarDecl>(ND)) ui.kind = "global_var";
                        else if (isa<TypedefDecl>(ND)) ui.kind = "typedef";
                        else if (isa<EnumConstantDecl>(ND)) ui.kind = "enum_constant";
                        else ui.kind = "decl";

                        SourceLocation DeclLoc = ND->getLocation();
                        ui.file_path = Analyzer.getFilePath(DeclLoc);
                        ui.start_line = Analyzer.getLineNumber(DeclLoc);
                        if (auto *FD = dyn_cast<FunctionDecl>(ND)) {
                            ui.end_line = Analyzer.getLineNumber(FD->getEndLoc());
                        } else {
                            ui.end_line = ui.start_line;
                        }
                    } else {
                        ui.kind = "unknown";
                        ui.definition = "unknown";
                        ui.pending_ast_resolve = false;
                    }
                }
            }

            // ---- Rebuild keys ----
            // Pending uses have been resolved, so the uses pattern may have changed
            std::map<MacroKey, MacroEntry> newGlobalMacros;
            for (auto &[oldKey, entry] : globalMacros) {
                UsesPattern newPattern;
                newPattern.uses = entry.uses;
                MacroKey newKey{oldKey.name, oldKey.definition_location, newPattern};

                auto it = newGlobalMacros.find(newKey);
                if (it != newGlobalMacros.end()) {
                    for (const auto &app : entry.appearances)
                        it->second.appearances.insert(app);
                } else {
                    newGlobalMacros[newKey] = entry;
                }
            }
            globalMacros = std::move(newGlobalMacros);
        }

        // ---- Phase 2: is_const determination (existing logic) ----
        const auto &UseSites = PP.getMacroUseSites();
        
        std::map<unsigned, bool> ConstResults;

        MacroConstEvalVisitor Visitor(Context, UseSites, ConstResults);
        Visitor.TraverseDecl(Context.getTranslationUnitDecl());
        
        std::lock_guard<std::mutex> lock(macrosMutex);

        llvm::errs() << "=== ConstResults keys ===\n";
        for (auto &Pair : ConstResults) {
            llvm::errs() << "  [" << Pair.first << "] = " << (Pair.second ? "true" : "false") << "\n";
        }
        llvm::errs() << "=== globalMacros keys (binden_test only) ===\n";
        for (auto &[key, entry] : globalMacros) {
            if (entry.file_path.find("binden_test") != std::string::npos) {
                llvm::errs() << "  [" << key.name << "|" << key.definition_location << "]\n";
            }
        }

        for (auto &Pair : ConstResults) {
            unsigned defRawLoc = Pair.first;
            SourceLocation DefLoc = SourceLocation::getFromRawEncoding(defRawLoc);
            std::string defLocStr = Analyzer.getOriginalLocationString(DefLoc);
            
            // Find the matching definition in globalMacros
            for (auto &[key, entry] : globalMacros) {
                if (key.definition_location == defLocStr) {
                    if (entry.kind != "macro_function") {
                        // If the AST traversal returns true, set true; if false, preserve a prior true from preliminary determination
                        if (Pair.second) {
                            entry.isConst = true;
                        }
                        // If Pair.second == false, do not overwrite a prior true from preliminary determination
                        //entry.isConst = Pair.second;
                    }
                    break;
                }
            }
        }
        
        llvm::errs() << "Constant evaluation done: " 
                     << ConstResults.size() << " macros evaluated\n";
    }
};

class MacroAction : public ASTFrontendAction {
public:
    std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, StringRef) override {
        {
            std::lock_guard<std::mutex> lock(macrosMutex);
            currentActiveMacro.clear();
        }
        
        return std::make_unique<MacroASTConsumer>(
            &CI.getSourceManager(), 
            CI.getPreprocessor()
        );
    }
};

std::string escapeJSON(const std::string &s) {
    std::string result;
    for (char c : s) {
        switch (c) {
            case '"': result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default: result += c;
        }
    }
    return result;
}

void printMacrosAsJSON() {
    std::cout << "{\n";
    std::cout << "  \"macros\": [\n";
    
    bool first = true;
    for (const auto &[key, info] : globalMacros) {
        
        if (!first) {
            std::cout << ",\n";
        }
        first = false;
        
        std::cout << "    {\n";
        std::cout << "      \"kind\": \"" << info.kind << "\",\n";
        std::cout << "      \"name\": \"" << escapeJSON(info.name) << "\",\n";
        std::cout << "      \"definition\": \"" << escapeJSON(info.definition_location) << "\",\n";
        std::cout << "      \"file_path\": \"" << escapeJSON(info.file_path) << "\",\n";
        std::cout << "      \"start_line\": " << info.start_line << ",\n";
        std::cout << "      \"end_line\": " << info.end_line << ",\n";
        std::cout << "      \"is_const\": " << (info.isConst ? "true" : "false") << ",\n";
        std::cout << "      \"is_independent\": null,\n";
        std::cout << "      \"is_flag\": null,\n";
        std::cout << "      \"is_guard\": false,\n";
        std::cout << "      \"is_guarded\": false,\n";
        std::cout << "      \"expanded_value\": \"" << escapeJSON(info.expanded_value) << "\",\n";
        
        // parameters
        std::cout << "      \"parameters\": [";
        bool firstParam = true;
        for (const auto &param : info.parameters) {
            if (!firstParam) std::cout << ", ";
            firstParam = false;
            std::cout << "\"" << escapeJSON(param) << "\"";
        }
        std::cout << "],\n";

        // #undef location (if exists)
        if (!info.undef_location.empty()) {
            std::cout << "      \"undef\": \"" << escapeJSON(info.undef_location) << "\",\n";
        }
        
        std::cout << "      \"appearances\": [\n";
        bool firstAppearance = true;
        for (const auto &appearance : info.appearances) {
            if (!firstAppearance) {
                std::cout << ",\n";
            }
            firstAppearance = false;
            std::cout << "        \"" << escapeJSON(appearance) << "\"";
        }
        std::cout << "\n      ],\n";

        std::cout << "      \"uses\": [\n";
        bool firstUse = true;
        for (const auto &ui : info.uses) {
            if (!firstUse) std::cout << ",\n";
            firstUse = false;
            std::cout << "        {\n";
            std::cout << "          \"kind\": \"" << escapeJSON(ui.kind) << "\",\n";
            std::cout << "          \"name\": \"" << escapeJSON(ui.name) << "\",\n";
            std::cout << "          \"usage_location\": \"" << escapeJSON(ui.usage_location) << "\",\n";
            std::cout << "          \"definition\": \"" << escapeJSON(ui.definition) << "\",\n";
            std::cout << "          \"start_line\": " << ui.start_line << ",\n";
            std::cout << "          \"end_line\": " << ui.end_line << ",\n";
            std::cout << "          \"file_path\": \"" << escapeJSON(ui.file_path) << "\"\n";
            std::cout << "        }";
        }
        std::cout << "\n      ]\n";
        
        std::cout << "    }";
    }
    
    std::cout << "\n  ]\n";
    std::cout << "}\n";
}


void addCustomIncludePaths(ClangTool &Tool) {
    Tool.appendArgumentsAdjuster(
      [](const CommandLineArguments &Args, StringRef Filename) -> CommandLineArguments {
        CommandLineArguments NewArgs = Args;
        
        std::vector<std::string> CommonArgs = {
          "-resource-dir=/usr/lib/llvm-19/lib/clang/19",
          "-w",
          "-Wno-incompatible-function-pointer-types",
          "-Wno-incompatible-pointer-types",
          "-fno-strict-aliasing",
          "-isystem/usr/lib/llvm-19/lib/clang/19/include",
        };
  
        if (Filename.ends_with(".cxx") || Filename.ends_with(".cpp") || 
            Filename.ends_with(".cc") || Filename.ends_with(".C")) {

          std::vector<std::string> CxxArgs = {
            "-isystem/usr/include/c++/11",
            "-isystem/usr/include/x86_64-linux-gnu/c++/11",
            "-isystem/usr/include/c++/11/backward",
          };
          CommonArgs.insert(CommonArgs.end(), CxxArgs.begin(), CxxArgs.end());
        }
  
        CommonArgs.push_back("-isystem/usr/include/x86_64-linux-gnu");
        CommonArgs.push_back("-isystem/usr/include");
  
        NewArgs.insert(NewArgs.begin() + 1, CommonArgs.begin(), CommonArgs.end());
        return NewArgs;
      }
    );
}

  
static llvm::cl::OptionCategory MyToolCategory("macro analyzer options");

int main(int argc, const char **argv) {
    
    if (argc >= 2 && std::string(argv[1]) != "-p") {
        std::vector<std::string> SourcePaths;
        SourcePaths.push_back(argv[1]);
        
        std::vector<std::string> CompileCommands;
        CompileCommands.insert(
            CompileCommands.end(),
            DEFAULT_INCLUDE_PATHS.begin(),
            DEFAULT_INCLUDE_PATHS.end()
        );
        
        bool afterDashes = false;
        for (int i = 2; i < argc; ++i) {
            if (std::string(argv[i]) == "--") {
                afterDashes = true;
                continue;
            }
            if (afterDashes) {
                CompileCommands.push_back(argv[i]);
            }
        }
        
        auto Compilations = std::make_unique<FixedCompilationDatabase>(
            ".", CompileCommands);
        
        ClangTool Tool(*Compilations, SourcePaths);
        int result = Tool.run(newFrontendActionFactory<MacroAction>().get());
        
        printMacrosAsJSON();
        return result;
    }
    
    if (argc >= 3 && std::string(argv[1]) == "-p") {
        std::string dirPath = argv[2];
        
        // added: setup compile_dir
        std::filesystem::path absDir = std::filesystem::absolute(dirPath);
        g_compileDir = absDir.lexically_normal().string();
        // ended
        
        std::string ErrorMessage;
        auto Compilations = CompilationDatabase::autoDetectFromDirectory(dirPath, ErrorMessage);
        
        if (!Compilations) {
            llvm::errs() << "Error: " << ErrorMessage << "\n";
            return 1;
        }
        
        auto AllFiles = Compilations->getAllFiles();
        if (AllFiles.empty()) {
            llvm::errs() << "No source files found in compile_commands.json\n";
            return 1;
        }
        
        ClangTool Tool(*Compilations, AllFiles);

        addCustomIncludePaths(Tool);

        int result = Tool.run(newFrontendActionFactory<MacroAction>().get());
        
        printMacrosAsJSON();
        return result;
    }
    
    auto ExpectedParser = CommonOptionsParser::create(argc, argv, MyToolCategory);
    if (!ExpectedParser) {
        llvm::errs() << ExpectedParser.takeError();
        return 1;
    }
    CommonOptionsParser &OptionsParser = ExpectedParser.get();
    ClangTool Tool(OptionsParser.getCompilations(), OptionsParser.getSourcePathList());
    
    int result = Tool.run(newFrontendActionFactory<MacroAction>().get());
    printMacrosAsJSON();
    
    return result;
}

/*
EvaluateAsRValue
*/