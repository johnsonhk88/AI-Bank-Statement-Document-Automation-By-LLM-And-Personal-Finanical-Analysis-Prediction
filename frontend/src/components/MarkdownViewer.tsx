import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface MarkdownViewerProps {
  content: string | null | undefined;
}

export default function MarkdownViewer({ content }: MarkdownViewerProps) {
  if (!content) {
    return (
      <p className="text-sm text-gray-400 italic">No report yet</p>
    );
  }

  return (
    <div className="text-sm text-gray-700 leading-relaxed max-w-none">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>
        {content}
      </ReactMarkdown>
    </div>
  );
}
