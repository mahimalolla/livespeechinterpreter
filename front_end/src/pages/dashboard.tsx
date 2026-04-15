import { useNavigate } from "react-router-dom";
import {
  Activity,
  Globe2,
  Stethoscope,
  Scale,
  MessageSquare,
  ShieldCheck,
  Lock,
  Mic,
  PlayCircle,
  Clock3,
  Languages,
  ChevronRight,
  FolderClock,
} from "lucide-react";
import vistaLogo from "@/assets/vista-logo.png";
import vistaLogoDark from "@/assets/vista-logo-dark.png";
import ThemeToggle from "@/components/ThemeToggle";
import { useEffect, useState } from "react";

const Dashboard = () => {
  const navigate = useNavigate();
  const [isDark, setIsDark] = useState(() =>
    document.documentElement.classList.contains("dark")
  );

  useEffect(() => {
    const observer = new MutationObserver(() => {
      setIsDark(document.documentElement.classList.contains("dark"));
    });
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["class"],
    });
    return () => observer.disconnect();
  }, []);

  const stats = [
    {
      label: "Active Session",
      value: "Live",
      icon: Activity,
      subtext: "Listening and translating",
    },
    {
      label: "Language Pair",
      value: "English → Spanish",
      icon: Languages,
      subtext: "Real-time interpretation",
    },
    {
      label: "Mode",
      value: "Medical",
      icon: Stethoscope,
      subtext: "Domain-aware translation",
    },
    {
      label: "Voice Output",
      value: "Enabled",
      icon: PlayCircle,
      subtext: "Translated speech playback",
    },
  ];

  const recentSessions = [
    {
      title: "Patient Intake Conversation",
      domain: "Medical",
      languages: "English → Spanish",
      duration: "12 min",
      icon: Stethoscope,
    },
    {
      title: "Contract Review Discussion",
      domain: "Legal",
      languages: "English → Spanish",
      duration: "18 min",
      icon: Scale,
    },
    {
      title: "General Help Desk Support",
      domain: "General",
      languages: "English → Spanish",
      duration: "9 min",
      icon: MessageSquare,
    },
  ];

  const quickActions = [
    {
      title: "Start New Session",
      desc: "Launch a new real-time translation session.",
      icon: Mic,
      action: () => navigate("/translate"),
    },
    {
      title: "View Live Interpreter",
      desc: "Open the active voice translation workspace.",
      icon: Globe2,
      action: () => navigate("/translate"),
    },
    {
      title: "Review Recent Sessions",
      desc: "Go through the latest translated conversations.",
      icon: FolderClock,
      action: () => {},
    },
  ];

  return (
    <div className="min-h-screen bg-background text-foreground relative">
      <div className="absolute inset-0 pointer-events-none bg-[radial-gradient(circle_at_top,rgba(255,140,0,0.10),transparent_35%)]" />

      {/* Navbar */}
      <nav className="sticky top-0 z-50 border-b border-white/10 bg-background/55 backdrop-blur-2xl supports-[backdrop-filter]:bg-background/45 shadow-[0_8px_30px_rgb(0,0,0,0.12)]">
        <div className="max-w-7xl mx-auto flex items-center justify-between px-6 md:px-8 py-4 md:py-5">
          <div className="flex items-center gap-3 md:gap-4">
            <div className="relative">
              <div className="absolute inset-0 rounded-full bg-primary/20 blur-lg" />
              <img
                src={isDark ? vistaLogoDark : vistaLogo}
                alt="VISTA"
                className="relative w-11 h-11 md:w-14 md:h-14 rounded-xl object-contain"
              />
            </div>
            <div className="flex flex-col leading-none">
              <span className="text-lg md:text-2xl font-extrabold tracking-tight text-foreground">
                VISTA
              </span>
              <span className="hidden md:block text-[11px] uppercase tracking-[0.22em] text-muted-foreground mt-1">
                Session Dashboard
              </span>
            </div>
          </div>

          <div className="flex items-center gap-3 md:gap-4">
            <ThemeToggle />
            <button
              onClick={() => navigate("/translate")}
              className="group inline-flex items-center gap-2 px-5 md:px-6 py-2.5 md:py-3 rounded-xl bg-primary text-primary-foreground text-sm md:text-base font-semibold transition-all duration-300 shadow-[0_0_25px_rgba(255,165,0,0.22)] hover:bg-primary/90 hover:scale-[1.03] active:scale-[0.98]"
            >
              Open Interpreter
              <ChevronRight className="w-4 h-4 transition-transform duration-300 group-hover:translate-x-0.5" />
            </button>
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-6 md:px-8 py-10 md:py-12 relative z-10">
        {/* Header */}
        <section className="mb-10">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-border bg-muted/50 text-xs text-muted-foreground mb-5">
            <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
            Live session overview
          </div>

          <h1 className="text-3xl md:text-5xl font-bold tracking-tight mb-3">
            VISTA Dashboard
          </h1>
          <p className="text-muted-foreground text-base md:text-lg max-w-3xl leading-relaxed">
            A clear view of your live interpretation workflow, recent sessions,
            supported domains, and privacy-first translation experience.
          </p>
        </section>

        {/* Top stats */}
        <section className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-5 mb-10">
          {stats.map((item) => (
            <div
              key={item.label}
              className="rounded-2xl border border-border/60 bg-card/80 backdrop-blur-sm p-6 shadow-sm hover:-translate-y-1 transition-all duration-300"
            >
              <div className="w-11 h-11 rounded-xl bg-primary/10 flex items-center justify-center mb-4">
                <item.icon className="w-5 h-5 text-primary" />
              </div>
              <p className="text-sm text-muted-foreground mb-1">{item.label}</p>
              <h3 className="text-xl md:text-2xl font-bold text-foreground">
                {item.value}
              </h3>
              <p className="text-sm text-muted-foreground mt-2">
                {item.subtext}
              </p>
            </div>
          ))}
        </section>

        {/* Main grid */}
        <section className="grid grid-cols-1 xl:grid-cols-3 gap-6 mb-10">
          {/* Live session card */}
          <div className="xl:col-span-2 rounded-2xl border border-border/60 bg-card/80 backdrop-blur-sm p-6">
            <div className="flex items-center justify-between mb-6 gap-4 flex-wrap">
              <div>
                <p className="text-sm text-muted-foreground mb-1">
                  Current Session
                </p>
                <h2 className="text-2xl font-bold tracking-tight">
                  Real-Time Interpretation Console
                </h2>
              </div>
              <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-sm font-medium">
                <Activity className="w-4 h-4" />
                Live
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
              <div className="rounded-xl border border-border/50 bg-background/40 p-4">
                <p className="text-sm text-muted-foreground mb-2">Source</p>
                <div className="flex items-center gap-2 text-foreground font-semibold">
                  <Languages className="w-4 h-4 text-primary" />
                  English
                </div>
              </div>

              <div className="rounded-xl border border-border/50 bg-background/40 p-4">
                <p className="text-sm text-muted-foreground mb-2">Target</p>
                <div className="flex items-center gap-2 text-foreground font-semibold">
                  <Globe2 className="w-4 h-4 text-primary" />
                  Spanish
                </div>
              </div>

              <div className="rounded-xl border border-border/50 bg-background/40 p-4">
                <p className="text-sm text-muted-foreground mb-2">Domain</p>
                <div className="flex items-center gap-2 text-foreground font-semibold">
                  <Stethoscope className="w-4 h-4 text-primary" />
                  Medical
                </div>
              </div>
            </div>

            <div className="rounded-2xl border border-border/50 bg-background/40 p-5">
              <p className="text-sm text-muted-foreground mb-3">
                Live Transcript Preview
              </p>
              <div className="space-y-3">
                <div className="rounded-xl bg-muted/40 border border-border/40 p-4">
                  <p className="text-sm text-foreground">
                    Hello, I need help explaining the treatment plan to the
                    patient.
                  </p>
                </div>
                <div className="rounded-xl bg-primary/10 border border-primary/20 p-4">
                  <p className="text-sm text-foreground">
                    Hola, necesito ayuda para explicar el plan de tratamiento al
                    paciente.
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Quick actions */}
          <div className="rounded-2xl border border-border/60 bg-card/80 backdrop-blur-sm p-6">
            <h2 className="text-xl font-bold tracking-tight mb-5">
              Quick Actions
            </h2>
            <div className="space-y-4">
              {quickActions.map((item) => (
                <button
                  key={item.title}
                  onClick={item.action}
                  className="w-full text-left rounded-xl border border-border/50 bg-background/40 p-4 hover:border-primary/30 hover:-translate-y-0.5 transition-all duration-300"
                >
                  <div className="flex items-start gap-3">
                    <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center shrink-0">
                      <item.icon className="w-5 h-5 text-primary" />
                    </div>
                    <div>
                      <h3 className="font-semibold text-foreground mb-1">
                        {item.title}
                      </h3>
                      <p className="text-sm text-muted-foreground">
                        {item.desc}
                      </p>
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </div>
        </section>

        {/* Lower grid */}
        <section className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Recent sessions */}
          <div className="lg:col-span-2 rounded-2xl border border-border/60 bg-card/80 backdrop-blur-sm p-6">
            <div className="flex items-center justify-between mb-5">
              <h2 className="text-xl font-bold tracking-tight">
                Recent Sessions
              </h2>
              <div className="text-sm text-muted-foreground">
                Last 3 conversations
              </div>
            </div>

            <div className="space-y-4">
              {recentSessions.map((session) => (
                <div
                  key={session.title}
                  className="rounded-xl border border-border/50 bg-background/40 p-4 flex flex-col md:flex-row md:items-center md:justify-between gap-4"
                >
                  <div className="flex items-start gap-3">
                    <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center shrink-0">
                      <session.icon className="w-5 h-5 text-primary" />
                    </div>
                    <div>
                      <h3 className="font-semibold text-foreground">
                        {session.title}
                      </h3>
                      <p className="text-sm text-muted-foreground mt-1">
                        {session.domain} • {session.languages}
                      </p>
                    </div>
                  </div>

                  <div className="inline-flex items-center gap-2 text-sm text-muted-foreground">
                    <Clock3 className="w-4 h-4" />
                    {session.duration}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Trust and compliance */}
          <div className="rounded-2xl border border-border/60 bg-card/80 backdrop-blur-sm p-6">
            <h2 className="text-xl font-bold tracking-tight mb-5">
              Trust & Privacy
            </h2>

            <div className="space-y-4">
              <div className="rounded-xl border border-border/50 bg-background/40 p-4">
                <div className="flex items-center gap-2 mb-2">
                  <ShieldCheck className="w-4 h-4 text-emerald-400" />
                  <span className="font-medium text-foreground">
                    HIPAA Ready
                  </span>
                </div>
                <p className="text-sm text-muted-foreground">
                  Designed with healthcare-focused privacy expectations in mind.
                </p>
              </div>

              <div className="rounded-xl border border-border/50 bg-background/40 p-4">
                <div className="flex items-center gap-2 mb-2">
                  <Lock className="w-4 h-4 text-blue-400" />
                  <span className="font-medium text-foreground">
                    GDPR Ready
                  </span>
                </div>
                <p className="text-sm text-muted-foreground">
                  Supports privacy-first workflows for global users and teams.
                </p>
              </div>

              <div className="rounded-xl border border-border/50 bg-background/40 p-4">
                <div className="flex items-center gap-2 mb-2">
                  <ShieldCheck className="w-4 h-4 text-primary" />
                  <span className="font-medium text-foreground">
                    Privacy First
                  </span>
                </div>
                <p className="text-sm text-muted-foreground">
                  Built to prioritize secure handling of interpreted sessions.
                </p>
              </div>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
};

export default Dashboard;
