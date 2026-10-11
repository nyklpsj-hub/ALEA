import { Workspace } from "@/components/Workspace";

export default async function SessionPage({
  params,
}: {
  params: Promise<{ sessionId: string }>;
}) {
  const { sessionId } = await params;
  return <Workspace sessionId={sessionId} />;
}
