// analyzer.cpp
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Index/USRGeneration.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Basic/FileEntry.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/SmallVector.h"
#include <iostream>
#include <map>
#include <set>
#include <mutex>

// C++17 filesystem check
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

// === Default include paths ===
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

struct SymbolInfo {
    std::string name;              // before expansion
    std::string expanded_name;     // after expansion
    std::string kind;
    std::string signature;
    std::string defLocation;
    std::string definitionRef;  // For declarations, the location of the corresponding definition
    unsigned startLine;
    unsigned endLine;
    std::set<std::string> uses;
    std::map<std::string, std::set<std::string>> usage_locations;  // USR -> set of location strings
};

// Global symbol map
static std::map<std::string, SymbolInfo> globalSymbols;
static std::mutex symbolsMutex;

// Struct for recording macro expansion info
struct MacroExpansionInfo {
    std::string macroName;
    std::string location;
    std::string ownerUSR;
};

// Temporarily store macro expansion info
static std::vector<MacroExpansionInfo> macroExpansions;
static std::mutex macroExpansionsMutex;

// Record macro dependencies
struct MacroDependency {
    std::string macroUSR;      // The expanded macro
    std::string refName;       // Name of the referenced identifier
    std::string location;      // Location where it was expanded
};

static std::vector<MacroDependency> macroDependencies;
static std::mutex macroDependenciesMutex;


class MyDependencyCollector {
public:
    ASTContext *Context;
    std::map<std::string, SymbolInfo> symbols;
    
private:
    std::string currentOwner;

public:
    // Replace around lines 98-114 of analyzer.cpp with the following
    std::string getOriginalName(const NamedDecl *D) {
        if (!D) return "";
        
        SourceManager &SM = Context->getSourceManager();
        SourceLocation Loc = D->getLocation();
        
        // llvm::errs() << "getOriginalName: Checking " << D->getNameAsString() 
        //             << " at Loc=" << Loc.getRawEncoding() 
        //             << " isMacroID=" << Loc.isMacroID() << "\n";
        
        if (Loc.isMacroID()) {
            FileID FID = SM.getFileID(Loc);
            unsigned Offset = SM.getFileOffset(Loc);
            
            // llvm::errs() << "  FID=" << FID.getHashValue() 
            //             << " Offset=" << Offset << "\n";
            
            std::string OriginalText = SM.getMacroTokenText(Loc);
            if (!OriginalText.empty()) {
                // llvm::errs() << "Found original name: " << OriginalText 
                //             << " (expanded: " << D->getNameAsString() << ")\n";
                return OriginalText;
            }
            // } else {
            //     llvm::errs() << "getMacroTokenText returned empty!\n";
            // }
        }
        
        return D->getNameAsString();
    }
    //

    explicit MyDependencyCollector(ASTContext *Context) : Context(Context) {}

    // std::string getUSR(const Decl *D) {
    //     if (!D) return "";
    //     llvm::SmallVector<char, 128> buf;
    //     if (!index::generateUSRForDecl(D, buf)) {
    //         return std::string(buf.data(), buf.size());
    //     }
    //     return "";
    // }

    std::string getUSR(const Decl *D) {
        if (!D) return "";
        llvm::SmallVector<char, 128> buf;
        if (!index::generateUSRForDecl(D, buf)) {
            std::string usr(buf.data(), buf.size());
    
            // Append the definition location to make it unique
            SourceManager &SM = Context->getSourceManager();
            SourceLocation Loc = D->getLocation();
            if (Loc.isValid()) {
                SourceLocation OriginalLoc = SM.getOriginalLoc(Loc);
                FileID FID = SM.getFileID(OriginalLoc);
                OptionalFileEntryRef FER = SM.getFileEntryRefForID(FID);
                if (FER) {
                    unsigned line = SM.getLineNumber(FID, SM.getFileOffset(OriginalLoc));
                    usr += "@" + getAbsolutePath(FER->getName()) + ":" + std::to_string(line);
                }
            }
    
            return usr;
        }
        return "";
    }
    
    // Path normalization function
    std::string getAbsolutePath(StringRef filename) {
        std::filesystem::path p(filename.str());
        
        std::error_code ec;
        if (!p.is_absolute()) {
            p = std::filesystem::absolute(p, ec);
        }
        
        return p.lexically_normal().string();
    }
    
    // Get the location before macro expansion
    std::string getLocationString(SourceLocation Loc) {
        if (Loc.isInvalid()) return "unknown";
        
        SourceManager &SM = Context->getSourceManager();
        
        // std::cerr << "DEBUG: Original Loc isMacroID = " << Loc.isMacroID() << std::endl;
        
        // Traverse the entire macro expansion chain to get the original location
        SourceLocation OriginalLoc = SM.getOriginalLoc(Loc);
        
        // std::cerr << "DEBUG: After getOriginalLoc isMacroID = " << OriginalLoc.isMacroID() << std::endl;
        // std::cerr << "DEBUG: Same location? " << (Loc.getRawEncoding() == OriginalLoc.getRawEncoding()) << std::endl;
        
        // Get file info directly from FileID
        FileID FID = SM.getFileID(OriginalLoc);
        
        // Use OptionalFileEntryRef (new API)
        OptionalFileEntryRef FER = SM.getFileEntryRefForID(FID);
        if (!FER) {
            // std::cerr << "DEBUG: FileEntry not found!" << std::endl;
            return "unknown";
        }
        // Get the actual file path
        std::string filePath = getAbsolutePath(FER->getName());
        
        unsigned line = SM.getLineNumber(FID, SM.getFileOffset(OriginalLoc));
        unsigned column = SM.getColumnNumber(FID, SM.getFileOffset(OriginalLoc));
        
        std::string Result;
        llvm::raw_string_ostream OS(Result);
        OS << filePath << ":" << line << ":" << column;
        
        // std::cerr << "DEBUG: Result = " << OS.str() << std::endl;
        
        return OS.str();
    }

    // Generate function signature
    std::string generateFunctionSignature(FunctionDecl *FD) {
        if (!FD) return "";
        
        std::string sig;
        
        // Return type
        sig += FD->getReturnType().getAsString();
        sig += " ";
        
        // Function name
        sig += FD->getNameAsString();
        sig += "(";
        
        // Parameters
        bool firstParam = true;
        for (auto *Param : FD->parameters()) {
            if (!firstParam) {
                sig += ", ";
            }
            firstParam = false;
            
            sig += Param->getType().getAsString();
            if (Param->getNameAsString() != "") {
                sig += " ";
                sig += Param->getNameAsString();
            }
        }
        
        sig += ")";
        return sig;
    }

    // Get the line number before macro expansion
    unsigned getLineNumber(SourceLocation Loc) {
        if (Loc.isInvalid()) return 0;
        SourceManager &SM = Context->getSourceManager();
        
        SourceLocation OriginalLoc = SM.getOriginalLoc(Loc);
        
        FileID FID = SM.getFileID(OriginalLoc);
        return SM.getLineNumber(FID, SM.getFileOffset(OriginalLoc));
    }

    void collectDepsFromStmt(Stmt *S) {
        if (!S || currentOwner.empty()) return;

        if (auto *DRE = dyn_cast<DeclRefExpr>(S)) {
            if (auto *D = DRE->getDecl()) {
                std::string depUSR = getUSR(D);
                if (!depUSR.empty() && depUSR != currentOwner) {
                    symbols[currentOwner].uses.insert(depUSR);
                    symbols[currentOwner].usage_locations[depUSR].insert(
                        getLocationString(DRE->getLocation())
                    );
                }
            }
        }

        // if (auto *CE = dyn_cast<CallExpr>(S)) {
        //     if (auto *Callee = CE->getDirectCallee()) {
        //         std::string depUSR = getUSR(Callee);
        //         if (!depUSR.empty() && depUSR != currentOwner) {
        //             symbols[currentOwner].uses.insert(depUSR);
        //             symbols[currentOwner].usage_locations[depUSR].insert(
        //                 getLocationString(CE->getBeginLoc())
        //             );
        //         }
        //     }
        // }

        if (auto *CE = dyn_cast<CallExpr>(S)) {
            if (auto *Callee = CE->getDirectCallee()) {
                if (FunctionDecl *Def = Callee->getDefinition()) {
                    Callee = Def;
                }
                std::string depUSR = getUSR(Callee);
                if (!depUSR.empty() && depUSR != currentOwner) {
                    symbols[currentOwner].uses.insert(depUSR);
                    symbols[currentOwner].usage_locations[depUSR].insert(
                        getLocationString(CE->getBeginLoc())
                    );
                }
            }
        }

        if (auto *ME = dyn_cast<MemberExpr>(S)) {
            if (auto *Field = ME->getMemberDecl()) {
                std::string depUSR = getUSR(Field);
                if (!depUSR.empty() && depUSR != currentOwner) {
                    symbols[currentOwner].uses.insert(depUSR);
                    symbols[currentOwner].usage_locations[depUSR].insert(
                        getLocationString(ME->getMemberLoc())
                    );
                }
            }
        }

        collectTypeRefs(S);

        for (auto *Child : S->children()) {
            collectDepsFromStmt(Child);
        }
    }

    void collectTypeRefs(Stmt *S) {
        if (!S) return;

        if (auto *DS = dyn_cast<DeclStmt>(S)) {
            for (auto *D : DS->decls()) {
                if (auto *VD = dyn_cast<VarDecl>(D)) {
                    collectTypeDeps(VD->getType(), VD->getLocation());
                }
            }
        }

        if (auto *CE = dyn_cast<CastExpr>(S)) {
            collectTypeDeps(CE->getType(), CE->getBeginLoc());
        }
    }

    void collectTypeDeps(QualType QT, SourceLocation UseLoc = SourceLocation()) {
        if (QT.isNull() || currentOwner.empty()) return;

        if (auto *TDT = QT->getAs<TypedefType>()) {
            std::string depUSR = getUSR(TDT->getDecl());
            if (!depUSR.empty()) {
                symbols[currentOwner].uses.insert(depUSR);
                if (UseLoc.isValid()) {
                    symbols[currentOwner].usage_locations[depUSR].insert(
                        getLocationString(UseLoc)
                    );
                }
            }
        }

        if (auto *RT = QT->getAs<RecordType>()) {
            std::string depUSR = getUSR(RT->getDecl());
            if (!depUSR.empty()) {
                symbols[currentOwner].uses.insert(depUSR);
                if (UseLoc.isValid()) {
                    symbols[currentOwner].usage_locations[depUSR].insert(
                        getLocationString(UseLoc)
                    );
                }
            }
        }

        if (QT->isPointerType()) {
            collectTypeDeps(QT->getPointeeType(), UseLoc);
        }

        if (QT->isArrayType()) {
            if (auto *AT = dyn_cast<ArrayType>(QT.getTypePtr())) {
                collectTypeDeps(AT->getElementType(), UseLoc);
            }
        }
    }

    void setCurrentOwner(const std::string &usr) {
        currentOwner = usr;
    }

    void clearCurrentOwner() {
        currentOwner = "";
    }
    
    std::string getCurrentOwner() const {
        return currentOwner;
    }
    
    // Merge into the global symbol map
    void mergeToGlobal() {
        std::lock_guard<std::mutex> lock(symbolsMutex);
        for (auto &pair : symbols) {
            globalSymbols[pair.first] = pair.second;
        }
    }
};


std::string classifyVarDecl(VarDecl *VD) {
    if (VD->isStaticLocal()) return "static_local_var";
    
    for (const DeclContext *DC = VD->getDeclContext(); DC; DC = DC->getParent()) {
        if (isa<FunctionDecl>(DC)) return "local_var";
    }
    
    //if (VD->hasGlobalStorage()) return "global_var";
    if (VD->hasGlobalStorage()) {
        if (VD->hasExternalStorage() && !VD->hasInit()) {
            return "global_var_decl";
        }
        return "global_var";
    }
    return "variable";
}


class SymbolVisitor : public RecursiveASTVisitor<SymbolVisitor> {
private:
    MyDependencyCollector &Collector;

    bool isSystemLocation(SourceLocation Loc) {
        if (Loc.isInvalid()) return true;
        
        // SourceManager &SM = Collector.Context->getSourceManager();
        // // Check system header using the location before macro expansion
        // SourceLocation OriginalLoc = SM.getOriginalLoc(Loc);
        // return SM.isInSystemHeader(OriginalLoc);
        return false;
    }

public:
    explicit SymbolVisitor(MyDependencyCollector &Collector) : Collector(Collector) {}

    bool VisitFunctionDecl(FunctionDecl *FD) {
        // if (!FD->hasBody()) return true;
        if (isSystemLocation(FD->getLocation())) return true;
    
        std::string usr = Collector.getUSR(FD);
        if (usr.empty()) return true;

        // added
        // llvm::errs() << "\n=== VisitFunctionDecl ===\n";
        // llvm::errs() << "Function name: " << FD->getNameAsString() << "\n";
        // llvm::errs() << "USR: " << usr << "\n";

        // SourceManager &SM = Collector.Context->getSourceManager();
        // SourceLocation BeginLoc = FD->getBeginLoc();
        // SourceLocation FuncNameLoc = FD->getLocation();
        
        // llvm::errs() << "BeginLoc raw: " << BeginLoc.getRawEncoding() << "\n";
        // llvm::errs() << "BeginLoc location: " << Collector.getLocationString(BeginLoc) << "\n";
        // llvm::errs() << "FuncNameLoc location: " << Collector.getLocationString(FuncNameLoc) << "\n";
        
        // ended

        if (!FD->hasBody()) {
            // Prototype declaration
            std::string declLocation = Collector.getLocationString(FD->getLocation());
            std::string declUSR = usr + "#decl@" + declLocation;

            Collector.symbols[declUSR].name = FD->getNameAsString();
            Collector.symbols[declUSR].expanded_name = FD->getNameAsString();
            Collector.symbols[declUSR].kind = "function_decl";
            Collector.symbols[declUSR].signature = Collector.generateFunctionSignature(FD);
            Collector.symbols[declUSR].defLocation = declLocation;
            Collector.symbols[declUSR].startLine = Collector.getLineNumber(FD->getBeginLoc());
            Collector.symbols[declUSR].endLine = Collector.getLineNumber(FD->getEndLoc());

            Collector.setCurrentOwner(declUSR);
            for (auto *Param : FD->parameters()) {
                Collector.collectTypeDeps(Param->getType(), Param->getLocation());
            }
            Collector.collectTypeDeps(FD->getReturnType(), FD->getLocation());
            Collector.clearCurrentOwner();

            return true;
        }

    
        // Use getMacroTokenText()
        std::string originalName = Collector.getOriginalName(FD);
        std::string expandedName = FD->getNameAsString();
        
        Collector.symbols[usr].name = originalName;           // Name before expansion
        Collector.symbols[usr].expanded_name = expandedName;  // Name after expansion
        Collector.symbols[usr].kind = "function";
        Collector.symbols[usr].signature = Collector.generateFunctionSignature(FD);
        Collector.symbols[usr].defLocation = Collector.getLocationString(FD->getLocation());
        //Collector.symbols[usr].startLine = Collector.getLineNumber(FD->getLocation());
        Collector.symbols[usr].startLine = Collector.getLineNumber(FD->getBeginLoc());
        Collector.symbols[usr].endLine = Collector.getLineNumber(FD->getEndLoc());
    
        Collector.setCurrentOwner(usr);
    
        for (auto *Param : FD->parameters()) {
            Collector.collectTypeDeps(Param->getType(), Param->getLocation());
        }
    
        Collector.collectTypeDeps(FD->getReturnType(), FD->getLocation());
    
        if (Stmt *Body = FD->getBody()) {
            Collector.collectDepsFromStmt(Body);
        }
    
        Collector.clearCurrentOwner();
        return true;
    }


    bool VisitVarDecl(VarDecl *VD) {
        if (isSystemLocation(VD->getLocation())) return true;
    
        std::string usr = Collector.getUSR(VD);
        if (usr.empty()) return true;
    
        std::string varKind = classifyVarDecl(VD);
    
        if (varKind == "global_var_decl") {
            // extern declaration: register with #decl@ suffixed USR, same as function_decl and struct_decl
            std::string declLocation = Collector.getLocationString(VD->getLocation());
            std::string declUSR = usr + "#decl@" + declLocation;
    
            Collector.symbols[declUSR].name = VD->getNameAsString();
            Collector.symbols[declUSR].kind = "global_var_decl";
            Collector.symbols[declUSR].defLocation = declLocation;
            Collector.symbols[declUSR].startLine = Collector.getLineNumber(VD->getLocation());
            Collector.symbols[declUSR].endLine = Collector.getLineNumber(VD->getEndLoc());
    
            Collector.setCurrentOwner(declUSR);
            Collector.collectTypeDeps(VD->getType(), VD->getLocation());
            Collector.clearCurrentOwner();
    
            return true;
        }
    
        Collector.symbols[usr].name = VD->getNameAsString();
        Collector.symbols[usr].kind = varKind;
        Collector.symbols[usr].defLocation = Collector.getLocationString(VD->getLocation());
        Collector.symbols[usr].startLine = Collector.getLineNumber(VD->getLocation());
        Collector.symbols[usr].endLine = Collector.getLineNumber(VD->getEndLoc());
    
        Collector.setCurrentOwner(usr);
        Collector.collectTypeDeps(VD->getType(), VD->getLocation());
    
        if (VD->hasInit()) {
            Collector.collectDepsFromStmt(VD->getInit());
        }
    
        Collector.clearCurrentOwner();
        return true;
    }

    // bool VisitVarDecl(VarDecl *VD) {
    //     if (isSystemLocation(VD->getLocation())) return true;

    //     std::string usr = Collector.getUSR(VD);
    //     if (usr.empty()) return true;
        
    //     Collector.symbols[usr].name = VD->getNameAsString();
    //     // Collector.symbols[usr].kind = VD->isLocalVarDecl() ? "local_var" : 
    //     //                                VD->isStaticLocal() ? "static_local_var" :
    //     //                                VD->isFileVarDecl() ? "global_var" : "variable";

    //     Collector.symbols[usr].kind = classifyVarDecl(VD);

    //     Collector.symbols[usr].defLocation = Collector.getLocationString(VD->getLocation());
    //     Collector.symbols[usr].startLine = Collector.getLineNumber(VD->getLocation());
    //     Collector.symbols[usr].endLine = Collector.getLineNumber(VD->getEndLoc());

    //     Collector.setCurrentOwner(usr);
    //     Collector.collectTypeDeps(VD->getType(), VD->getLocation());

    //     if (VD->hasInit()) {
    //         Collector.collectDepsFromStmt(VD->getInit());
    //     }

    //     Collector.clearCurrentOwner();
    //     return true;
    // }

    bool VisitRecordDecl(RecordDecl *RD) {
        // if (!RD->isCompleteDefinition()) return true;
        if (isSystemLocation(RD->getLocation())) return true;

        std::string usr = Collector.getUSR(RD);
        if (usr.empty()) return true;

        // Collector.symbols[usr].name = RD->getNameAsString();
        // Collector.symbols[usr].kind = RD->isStruct() ? "struct" : 
        //                                RD->isUnion() ? "union" : "record";

        
        std::string baseKind = RD->isStruct() ? "struct" : 
                               RD->isUnion() ? "union" : "record";

        if (!RD->isCompleteDefinition()) {
            // Forward declaration
            std::string declLocation = Collector.getLocationString(RD->getLocation());
            std::string declUSR = usr + "#decl@" + declLocation;

            Collector.symbols[declUSR].name = RD->getNameAsString();
            Collector.symbols[declUSR].kind = baseKind + "_decl";
            Collector.symbols[declUSR].defLocation = declLocation;
            Collector.symbols[declUSR].startLine = Collector.getLineNumber(RD->getLocation());
            Collector.symbols[declUSR].endLine = Collector.getLineNumber(RD->getEndLoc());

            return true;
        }

        // Complete definition
        Collector.symbols[usr].name = RD->getNameAsString();
        Collector.symbols[usr].kind = baseKind;
        Collector.symbols[usr].defLocation = Collector.getLocationString(RD->getLocation());
        Collector.symbols[usr].startLine = Collector.getLineNumber(RD->getLocation());
        Collector.symbols[usr].endLine = Collector.getLineNumber(RD->getEndLoc());

        // Collector.symbols[usr].defLocation = Collector.getLocationString(RD->getLocation());
        // Collector.symbols[usr].startLine = Collector.getLineNumber(RD->getLocation());
        // Collector.symbols[usr].endLine = Collector.getLineNumber(RD->getEndLoc());

        Collector.setCurrentOwner(usr);

        for (auto *Field : RD->fields()) {
            Collector.collectTypeDeps(Field->getType(), Field->getLocation());
        }

        Collector.clearCurrentOwner();
        return true;
    }

    bool VisitTypedefDecl(TypedefDecl *TD) {
        if (isSystemLocation(TD->getLocation())) return true;

        std::string usr = Collector.getUSR(TD);
        if (usr.empty()) return true;

        Collector.symbols[usr].name = TD->getNameAsString();
        Collector.symbols[usr].kind = "typedef";
        Collector.symbols[usr].defLocation = Collector.getLocationString(TD->getLocation());
        Collector.symbols[usr].startLine = Collector.getLineNumber(TD->getLocation());
        Collector.symbols[usr].endLine = Collector.getLineNumber(TD->getEndLoc());

        Collector.setCurrentOwner(usr);
        Collector.collectTypeDeps(TD->getUnderlyingType(), TD->getLocation());
        Collector.clearCurrentOwner();

        return true;
    }

    bool VisitEnumDecl(EnumDecl *ED) {
        // if (!ED->isComplete()) return true;
        if (isSystemLocation(ED->getLocation())) return true;

        std::string usr = Collector.getUSR(ED);
        if (usr.empty()) return true;

        //
        if (!ED->isComplete()) {
            // Forward declaration
            std::string declLocation = Collector.getLocationString(ED->getLocation());
            std::string declUSR = usr + "#decl@" + declLocation;

            Collector.symbols[declUSR].name = ED->getNameAsString();
            Collector.symbols[declUSR].kind = "enum_decl";
            Collector.symbols[declUSR].defLocation = declLocation;
            Collector.symbols[declUSR].startLine = Collector.getLineNumber(ED->getLocation());
            Collector.symbols[declUSR].endLine = Collector.getLineNumber(ED->getEndLoc());

            return true;
        }


        Collector.symbols[usr].name = ED->getNameAsString();
        Collector.symbols[usr].kind = "enum";
        Collector.symbols[usr].defLocation = Collector.getLocationString(ED->getLocation());
        Collector.symbols[usr].startLine = Collector.getLineNumber(ED->getLocation());
        Collector.symbols[usr].endLine = Collector.getLineNumber(ED->getEndLoc());

        return true;
    }

    bool VisitEnumConstantDecl(EnumConstantDecl *ECD) {
        if (isSystemLocation(ECD->getLocation())) return true;

        std::string usr = Collector.getUSR(ECD);
        if (usr.empty()) return true;

        Collector.symbols[usr].name = ECD->getNameAsString();
        Collector.symbols[usr].kind = "enum_constant";
        Collector.symbols[usr].defLocation = Collector.getLocationString(ECD->getLocation());
        Collector.symbols[usr].startLine = Collector.getLineNumber(ECD->getLocation());
        Collector.symbols[usr].endLine = Collector.getLineNumber(ECD->getEndLoc());

        Collector.setCurrentOwner(usr);
        if (ECD->getInitExpr()) {
            Collector.collectDepsFromStmt(ECD->getInitExpr());
        }
        Collector.clearCurrentOwner();

        return true;
    }

    bool VisitFieldDecl(FieldDecl *FD) {
        if (isSystemLocation(FD->getLocation())) return true;

        std::string usr = Collector.getUSR(FD);
        if (usr.empty()) return true;

        Collector.symbols[usr].name = FD->getNameAsString();
        Collector.symbols[usr].kind = "field";
        Collector.symbols[usr].defLocation = Collector.getLocationString(FD->getLocation());
        Collector.symbols[usr].startLine = Collector.getLineNumber(FD->getLocation());
        Collector.symbols[usr].endLine = Collector.getLineNumber(FD->getEndLoc());

        Collector.setCurrentOwner(usr);
        Collector.collectTypeDeps(FD->getType(), FD->getLocation());
        Collector.clearCurrentOwner();

        return true;
    }
};

class MacroCallback : public PPCallbacks {
private:
    MyDependencyCollector &Collector;
    Preprocessor &PP;

    bool isSystemLocation(SourceLocation Loc) {
        if (Loc.isInvalid()) return true;
        
        // SourceManager &SM = Collector.Context->getSourceManager();
        // // Check system header using the location before macro expansion
        // SourceLocation OriginalLoc = SM.getOriginalLoc(Loc);
        // return SM.isInSystemHeader(OriginalLoc);
        return false; 
    }

public:
    MacroCallback(MyDependencyCollector &Collector, Preprocessor &PP) 
        : Collector(Collector), PP(PP) {}

    void MacroDefined(const Token &MacroNameTok, const MacroDirective *MD) override {
        SourceLocation MacroLoc = MacroNameTok.getLocation();
        if (isSystemLocation(MacroLoc)) return;

        // std::string macroName = MacroNameTok.getIdentifierInfo()->getName().str();
        // std::string usr = "macro:" + macroName;

        std::string macroName = MacroNameTok.getIdentifierInfo()->getName().str();
        std::string location = Collector.getLocationString(MacroNameTok.getLocation());
        std::string usr = "macro:" + macroName + "@" + location;

        // llvm::errs() << "\n=== MacroDefined ===\n";
        // llvm::errs() << "Macro: " << macroName << "\n";
        // llvm::errs() << "Location: " << Collector.getLocationString(MacroLoc) << "\n";
        
        
        Collector.symbols[usr].name = macroName;
        Collector.symbols[usr].kind = "macro";
        Collector.symbols[usr].defLocation = Collector.getLocationString(MacroNameTok.getLocation());
        Collector.symbols[usr].startLine = Collector.getLineNumber(MacroNameTok.getLocation());
        Collector.symbols[usr].endLine = Collector.getLineNumber(MacroNameTok.getLocation());

        // Record dependencies in the macro body
        const MacroInfo *MI = MD->getMacroInfo();
        if (MI) {
            std::string defLocation = Collector.getLocationString(MacroNameTok.getLocation());
            
            for (const Token &Tok : MI->tokens()) {
                if (Tok.is(tok::identifier)) {
                    const IdentifierInfo *II = Tok.getIdentifierInfo();
                    if (!II) continue;
                    
                    std::string refName = II->getName().str();
                    
                    // Exclude macro parameters
                    bool isParam = false;
                    if (MI->isFunctionLike()) {
                        for (const IdentifierInfo *Param : MI->params()) {
                            if (Param && Param->getName() == refName) {
                                isParam = true;
                                break;
                            }
                        }
                    }
                    
                    if (!isParam) {
                        std::lock_guard<std::mutex> lock(macroDependenciesMutex);
                        macroDependencies.push_back({usr, refName, defLocation});
                    }
                }
            }
        }
    }

    void MacroExpands(const Token &MacroNameTok, const MacroDefinition &MD,
                 SourceRange Range, const MacroArgs *Args) override {

        SourceLocation MacroLoc = MacroNameTok.getLocation();
        
        if (isSystemLocation(MacroLoc)) return;

        // std::string macroName = MacroNameTok.getIdentifierInfo()->getName().str();
        // std::string macroUSR = "macro:" + macroName;
        // std::string location = Collector.getLocationString(MacroLoc);
        // std::string ownerUSR = Collector.getCurrentOwner();

        // // llvm::errs() << "\n=== MacroExpands ===\n";
        // // llvm::errs() << "Macro: " << macroName << "\n";
        // // llvm::errs() << "Location: " << Collector.getLocationString(MacroLoc) << "\n";
        // // llvm::errs() << "CurrentOwner: " << Collector.getCurrentOwner() << "\n";
    
        
        // {
        //     std::lock_guard<std::mutex> lock(macroExpansionsMutex);
        //     macroExpansions.push_back({macroName, location, ownerUSR});
        // }
        
        ///
        std::string macroName = MacroNameTok.getIdentifierInfo()->getName().str();
        std::string macroDefLocation = "";
        const MacroInfo *MI = MD.getMacroInfo();
        if (MI) {
            macroDefLocation = Collector.getLocationString(MI->getDefinitionLoc());
        }
        std::string macroUSR = "macro:" + macroName + "@" + macroDefLocation;

        std::string location = Collector.getLocationString(MacroLoc);
        std::string ownerUSR = Collector.getCurrentOwner();

        {
            std::lock_guard<std::mutex> lock(macroExpansionsMutex);
            macroExpansions.push_back({macroUSR, location, ownerUSR});
        }

        // const MacroInfo *MI = MD.getMacroInfo();
        // if (MI) {
        //     std::lock_guard<std::mutex> lock(macroDependenciesMutex);
            
        //     for (const Token &Tok : MI->tokens()) {
        //         if (Tok.is(tok::identifier)) {
        //             const IdentifierInfo *II = Tok.getIdentifierInfo();
        //             if (!II) continue;
                    
        //             std::string refName = II->getName().str();
                    
        //             bool isParam = false;
        //             if (MI->isFunctionLike()) {
        //                 for (const IdentifierInfo *Param : MI->params()) {
        //                     if (Param && Param->getName() == refName) {
        //                         isParam = true;
        //                         break;
        //                     }
        //                 }
        //             }
                    
        //             if (!isParam) {
        //                 macroDependencies.push_back({macroUSR, refName, location});
        //             }
        //         }
        //     }
        // }
    }
};

class SymbolConsumer : public ASTConsumer {
private:
    MyDependencyCollector Collector;
    SymbolVisitor Visitor;

public:
    explicit SymbolConsumer(ASTContext *Context, Preprocessor &PP) 
        : Collector(Context), Visitor(Collector) {
        PP.addPPCallbacks(std::make_unique<MacroCallback>(Collector, PP));
    }

    void HandleTranslationUnit(ASTContext &Context) override {
        Visitor.TraverseDecl(Context.getTranslationUnitDecl());
        
        // 1. Process existing macro expansion info
        {
            std::lock_guard<std::mutex> lock(macroExpansionsMutex);
            for (const auto &expansion : macroExpansions) {
                std::string macroUSR = expansion.macroName;  // Already a location-qualified USR
                if (!expansion.ownerUSR.empty() && Collector.symbols.count(macroUSR)) {
                    Collector.symbols[expansion.ownerUSR].uses.insert(macroUSR);
                    Collector.symbols[expansion.ownerUSR].usage_locations[macroUSR].insert(expansion.location);
                }
            }

            // for (const auto &expansion : macroExpansions) {
            //     std::string macroUSR = "macro:" + expansion.macroName;
            //     if (!expansion.ownerUSR.empty() && Collector.symbols.count(macroUSR)) {
            //         Collector.symbols[expansion.ownerUSR].uses.insert(macroUSR);
            //         Collector.symbols[expansion.ownerUSR].usage_locations[macroUSR].insert(expansion.location);
            //     }
            // }
            macroExpansions.clear();
        }
        
        // 2. Resolve macro dependencies
        {
            std::lock_guard<std::mutex> lock(macroDependenciesMutex);
            
            for (const auto &dep : macroDependencies) {
                // Check if the macro exists
                if (!Collector.symbols.count(dep.macroUSR)) continue;
                
                // Find the reference target
                std::string targetUSR;
                
                // 2-1. Look up as a macro
                // std::string macroTargetUSR = "macro:" + dep.refName;
                // if (Collector.symbols.count(macroTargetUSR)) {
                //     targetUSR = macroTargetUSR;
                // }

                std::string macroPrefix = "macro:" + dep.refName + "@";
                for (const auto &symbolPair : Collector.symbols) {
                    if (symbolPair.first.rfind(macroPrefix, 0) == 0) {
                        targetUSR = symbolPair.first;
                        break;
                    }
                }
                // 2-2. Look up as a function, variable, or type
                // else {
                if (targetUSR.empty()) {
                    for (const auto &symbolPair : Collector.symbols) {
                        const SymbolInfo &info = symbolPair.second;
                        
                        // Find a symbol with a matching name
                        if (info.name == dep.refName || 
                            info.expanded_name == dep.refName) {
                            targetUSR = symbolPair.first;
                            break;
                        }
                    }
                }
                
                // Record the dependency
                if (!targetUSR.empty()) {
                    Collector.symbols[dep.macroUSR].uses.insert(targetUSR);
                    Collector.symbols[dep.macroUSR].usage_locations[targetUSR].insert(dep.location);
                }
            }
            
            macroDependencies.clear();
        }

        // Link declarations to their corresponding definition locations
        for (auto &pair : Collector.symbols) {
            const std::string &key = pair.first;
            SymbolInfo &info = pair.second;

            // Entries containing "#decl@" are declarations
            size_t declPos = key.find("#decl@");
            if (declPos == std::string::npos) continue;

            // Extract the base USR (before "#decl@")
            std::string baseUSR = key.substr(0, declPos);

            // Check if a definition (without "#decl@") exists with the same base USR
            auto defIt = Collector.symbols.find(baseUSR);
            if (defIt != Collector.symbols.end()) {
                info.definitionRef = defIt->second.defLocation;
            }
        }
        //
        
        Collector.mergeToGlobal();
    }
};

class SymbolAction : public ASTFrontendAction {
public:
    std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, StringRef) override {
        return std::make_unique<SymbolConsumer>(&CI.getASTContext(), CI.getPreprocessor());
    }
};


std::string escapeJsonString(const std::string &str) {
    std::string result;
    result.reserve(str.size() * 2);
    
    for (char c : str) {
        switch (c) {
            case '"':  result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\n': result += "\\n";  break;
            case '\r': result += "\\r";  break;
            case '\t': result += "\\t";  break;
            case '\b': result += "\\b";  break;
            case '\f': result += "\\f";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
                    result += buf;
                } else {
                    result += c;
                }
        }
    }
    return result;
}



void printAllSymbols() {
    // // added 
    for (auto &pair : globalSymbols) {
        const std::string &key = pair.first;
        SymbolInfo &info = pair.second;
        
        size_t declPos = key.find("#decl@");
        if (declPos == std::string::npos) continue;
        
        std::string baseUSR = key.substr(0, declPos);
        auto defIt = globalSymbols.find(baseUSR);
        if (defIt != globalSymbols.end()) {
            info.definitionRef = defIt->second.defLocation;
        } else {
            std::string clangUSR = baseUSR;
            size_t pathAt = clangUSR.find("@/");
            if (pathAt != std::string::npos) {
                clangUSR = clangUSR.substr(0, pathAt);
            }
            
            std::vector<const SymbolInfo*> candidates;
            for (const auto &s : globalSymbols) {
                if (s.first.find("#decl@") != std::string::npos) continue;
                std::string sClangUSR = s.first;
                size_t sPathAt = sClangUSR.find("@/");
                if (sPathAt != std::string::npos) {
                    sClangUSR = sClangUSR.substr(0, sPathAt);
                }
                if (sClangUSR == clangUSR) {
                    candidates.push_back(&s.second);
                }
            }
            
            if (candidates.size() == 1) {
                info.definitionRef = candidates[0]->defLocation;
            }
        }
    }

    // for (auto &pair : globalSymbols) {
    //     const std::string &key = pair.first;
    //     SymbolInfo &info = pair.second;
        
    //     size_t declPos = key.find("#decl@");
    //     if (declPos == std::string::npos) continue;
        
    //     std::string baseUSR = key.substr(0, declPos);
    //     auto defIt = globalSymbols.find(baseUSR);

    //     llvm::errs() << "LINK: key=" << key << "\n";
    //     llvm::errs() << "LINK: baseUSR=" << baseUSR << "\n";
    //     llvm::errs() << "LINK: found=" << (defIt != globalSymbols.end()) << "\n";
        
    //     if (defIt != globalSymbols.end()) {
    //         info.definitionRef = defIt->second.defLocation;
    //         llvm::errs() << "LINK: set definitionRef=" << info.definitionRef << "\n";
    //     }
    // }
    // // ended
    
    std::cout << "{\n";
    std::cout << "  \"symbols\": [\n";
    
    bool first = true;
    for (auto it = globalSymbols.begin(); it != globalSymbols.end(); ++it) {
        const SymbolInfo &info = it->second;
        
        if (info.kind == "local_var" || info.kind == "static_local_var") continue;
        
        if (!first) {
            std::cout << ",\n";
        }
        first = false;
        
        std::cout << "    {\n";
        std::cout << "      \"kind\": \"" << escapeJsonString(info.kind) << "\",\n";
        std::cout << "      \"name\": \"" << escapeJsonString(info.name) << "\",\n";
        
        if (!info.expanded_name.empty() && info.expanded_name != info.name) {
            std::cout << "      \"expanded_name\": \"" << escapeJsonString(info.expanded_name) << "\",\n";
        }
        
        if (info.kind == "function" && !info.signature.empty()) {
            std::cout << "      \"signature\": \"" << escapeJsonString(info.signature) << "\",\n";
        }
        
        std::string filePath = info.defLocation;
        size_t lastColon = filePath.rfind(':');
        if (lastColon != std::string::npos) {
            size_t secondLastColon = filePath.rfind(':', lastColon - 1);
            if (secondLastColon != std::string::npos) {
                filePath = filePath.substr(0, secondLastColon);
            }
        }

        std::cout << "      \"definition\": \"" << escapeJsonString(info.defLocation) << "\",\n";
        if (!info.definitionRef.empty()) {
            std::cout << "      \"definition_ref\": \"" << escapeJsonString(info.definitionRef) << "\",\n";
        }
        std::cout << "      \"file_path\": \"" << escapeJsonString(filePath) << "\",\n";
        std::cout << "      \"start_line\": " << info.startLine << ",\n";
        std::cout << "      \"end_line\": " << info.endLine << ",\n";
        std::cout << "      \"uses\": [\n";
        
        bool firstDep = true;
        for (const auto &depUSR : info.uses) {

            //const SymbolInfo &dep = globalSymbols[depUSR];
            const SymbolInfo *dep = nullptr;
            if (globalSymbols.count(depUSR)) {
                dep = &globalSymbols[depUSR];
            } else {
                std::string prefix = depUSR + "#decl@";
                for (const auto &s : globalSymbols) {
                    if (s.first.rfind(prefix, 0) == 0) {
                        dep = &s.second;
                        break;
                    }
                }
            }
            if (!dep) continue;
            
            auto usageIt = info.usage_locations.find(depUSR);
            if (usageIt != info.usage_locations.end() && !usageIt->second.empty()) {
                for (const auto &loc : usageIt->second) {
                    if (!firstDep) {
                        std::cout << ",\n";
                    }
                    firstDep = false;
                    
                    std::cout << "        {\n";
                    // std::cout << "          \"kind\": \"" << escapeJsonString(dep.kind) << "\",\n";
                    // std::cout << "          \"name\": \"" << escapeJsonString(dep.name) << "\",\n";
                    // std::cout << "          \"definition\": \"" << escapeJsonString(dep.defLocation) << "\",\n";
                    // std::cout << "          \"usage_location\": \"" << escapeJsonString(loc) << "\"\n";
                    std::cout << "          \"kind\": \"" << escapeJsonString(dep->kind) << "\",\n";
                    std::cout << "          \"name\": \"" << escapeJsonString(dep->name) << "\",\n";
                    //std::cout << "          \"definition\": \"" << escapeJsonString(dep->defLocation) << "\",\n";
                    std::string defLocation = (!dep->definitionRef.empty()) ? dep->definitionRef : dep->defLocation;
                    std::cout << "          \"definition\": \"" << escapeJsonString(defLocation) << "\",\n";
                    std::cout << "          \"usage_location\": \"" << escapeJsonString(loc) << "\"\n";
                    std::cout << "        }";
                }
            } else {
                if (!firstDep) {
                    std::cout << ",\n";
                }
                firstDep = false;
                
                std::cout << "        {\n";
                std::cout << "          \"kind\": \"" << escapeJsonString(dep->kind) << "\",\n";
                std::cout << "          \"name\": \"" << escapeJsonString(dep->name) << "\",\n";
                //std::cout << "          \"definition\": \"" << escapeJsonString(dep->defLocation) << "\"\n";
                std::string defLocation = (!dep->definitionRef.empty()) ? dep->definitionRef : dep->defLocation;
                std::cout << "          \"definition\": \"" << escapeJsonString(defLocation) << "\"\n";
                // std::cout << "          \"kind\": \"" << escapeJsonString(dep.kind) << "\",\n";
                // std::cout << "          \"name\": \"" << escapeJsonString(dep.name) << "\",\n";
                // std::cout << "          \"definition\": \"" << escapeJsonString(dep.defLocation) << "\"\n";
                std::cout << "        }";

            }
        }
        
        std::cout << "\n      ]\n";
        std::cout << "    }";
    }
    
    std::cout << "\n  ]\n";
    std::cout << "}\n";
}


void printAllSymbols0() {
    std::cout << "{\n";
    std::cout << "  \"symbols\": [\n";
    
    bool first = true;
    for (auto it = globalSymbols.begin(); it != globalSymbols.end(); ++it) {
        const SymbolInfo &info = it->second;
        
        if (!first) {
            std::cout << ",\n";
        }
        first = false;
        
        std::cout << "    {\n";
        std::cout << "      \"kind\": \"" << info.kind << "\",\n";
        std::cout << "      \"name\": \"" << info.name << "\",\n";
        
        if (!info.expanded_name.empty() && info.expanded_name != info.name) {
            std::cout << "      \"expanded_name\": \"" << info.expanded_name << "\",\n";
        }
        
        if (info.kind == "function" && !info.signature.empty()) {
            std::cout << "      \"signature\": \"" << info.signature << "\",\n";
        }
        
        std::string filePath = info.defLocation;
        size_t lastColon = filePath.rfind(':');
        if (lastColon != std::string::npos) {
            size_t secondLastColon = filePath.rfind(':', lastColon - 1);
            if (secondLastColon != std::string::npos) {
                filePath = filePath.substr(0, secondLastColon);
            }
        }

        std::cout << "      \"definition\": \"" << info.defLocation << "\",\n";
        std::cout << "      \"file_path\": \"" << filePath << "\",\n";
        std::cout << "      \"start_line\": " << info.startLine << ",\n";
        std::cout << "      \"end_line\": " << info.endLine << ",\n";
        std::cout << "      \"uses\": [\n";
        
        bool firstDep = true;
        for (const auto &depUSR : info.uses) {
            if (globalSymbols.count(depUSR)) {
                const SymbolInfo &dep = globalSymbols[depUSR];
                
                auto usageIt = info.usage_locations.find(depUSR);
                if (usageIt != info.usage_locations.end() && !usageIt->second.empty()) {
                    for (const auto &loc : usageIt->second) {
                        if (!firstDep) {
                            std::cout << ",\n";
                        }
                        firstDep = false;
                        
                        std::cout << "        {\n";
                        std::cout << "          \"kind\": \"" << dep.kind << "\",\n";
                        std::cout << "          \"name\": \"" << dep.name << "\",\n";
                        std::cout << "          \"definition\": \"" << dep.defLocation << "\",\n";
                        std::cout << "          \"usage_location\": \"" << loc << "\"\n";
                        std::cout << "        }";
                    }
                } else {
                    if (!firstDep) {
                        std::cout << ",\n";
                    }
                    firstDep = false;
                    
                    std::cout << "        {\n";
                    std::cout << "          \"kind\": \"" << dep.kind << "\",\n";
                    std::cout << "          \"name\": \"" << dep.name << "\",\n";
                    std::cout << "          \"definition\": \"" << dep.defLocation << "\"\n";
                    std::cout << "        }";
                }
            }
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
          CommonArgs.push_back("-isystem/usr/include/c++/11");
          CommonArgs.push_back("-isystem/usr/include/x86_64-linux-gnu/c++/11");
          CommonArgs.push_back("-isystem/usr/include/c++/11/backward");
        }
  
        CommonArgs.push_back("-isystem/usr/include/x86_64-linux-gnu");
        CommonArgs.push_back("-isystem/usr/include");
  
        NewArgs.insert(NewArgs.begin() + 1, CommonArgs.begin(), CommonArgs.end());
        return NewArgs;
      }
    );
}

static llvm::cl::OptionCategory MyToolCategory("analyzer options");

int main(int argc, const char **argv) {
    std::vector<const char*> args;
    std::string dirPath;
    bool isDirectory = false;
    

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
        
        // create FixedCompilationDatabase
        auto Compilations = std::make_unique<FixedCompilationDatabase>(
            ".", CompileCommands);
        
        ClangTool Tool(*Compilations, SourcePaths);
        int result = Tool.run(newFrontendActionFactory<SymbolAction>().get());
        
        printAllSymbols();
        return result;
    }
    
    if (argc >= 3 && std::string(argv[1]) == "-p") {
        dirPath = argv[2];
        isDirectory = true;
        
        std::string ErrorMessage;
        auto Compilations = CompilationDatabase::autoDetectFromDirectory(dirPath, ErrorMessage);
        
        if (!Compilations) {
            // llvm::errs() << "Error: " << ErrorMessage << "\n";
            return 1;
        }
        
        auto AllFiles = Compilations->getAllFiles();
        if (AllFiles.empty()) {
            // llvm::errs() << "No source files found in compile_commands.json\n";
            return 1;
        }
        
        //int result = Tool.run(newFrontendActionFactory<SymbolAction>().get());
        ClangTool Tool(*Compilations, AllFiles);

        // Tool.appendArgumentsAdjuster(
        //     getInsertArgumentAdjuster("-resource-dir=/usr/lib/llvm-19/lib/clang/19", ArgumentInsertPosition::BEGIN));
        // Tool.appendArgumentsAdjuster(
        //     getInsertArgumentAdjuster("-w", ArgumentInsertPosition::END));
        
        addCustomIncludePaths(Tool);

        // for (const auto &arg : DEFAULT_INCLUDE_PATHS) {
        //     Tool.appendArgumentsAdjuster(
        //         getInsertArgumentAdjuster(arg.c_str(), ArgumentInsertPosition::BEGIN));
        //         //getInsertArgumentAdjuster(arg.c_str(), ArgumentInsertPosition::END));
        // }
        int result = Tool.run(newFrontendActionFactory<SymbolAction>().get());

        
        printAllSymbols();
        return result;
    }
    
    // CommonOptionsParser
    auto ExpectedParser = CommonOptionsParser::create(argc, argv, MyToolCategory);
    if (!ExpectedParser) {
        // llvm::errs() << ExpectedParser.takeError();
        return 1;
    }
    CommonOptionsParser &OptionsParser = ExpectedParser.get();
    ClangTool Tool(OptionsParser.getCompilations(), OptionsParser.getSourcePathList());
    
    int result = Tool.run(newFrontendActionFactory<SymbolAction>().get());
    printAllSymbols();
    
    return result;
}